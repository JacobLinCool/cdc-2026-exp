//! Exact teacher preferences between quiet public successor positions.
use cdc_engine::{
    board::*,
    eval::{evaluate, Params},
    game::*,
    neural::ValueNet,
    search::{LearnedEval, SearchConfig, Searcher},
    structured::{features, FEATURES},
};
use std::{
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

fn quiet_successor(pos: &Pos, mv: Move, history: &[u64]) -> Option<Pos> {
    if mv.is_flip() {
        return None;
    }
    let child = pos.make_move(mv);
    if child.nocap >= 50
        || history.contains(&child.hash)
        || child.alive_color(0) == 0
        || child.alive_color(1) == 0
    {
        return None;
    }
    if !child.has_moves() {
        return None;
    }
    let mut moves = MoveList::new();
    child.gen_captures(&mut moves);
    if moves.len != 0 {
        return None;
    }
    let mut opponent = child.clone();
    opponent.side ^= 1;
    opponent.gen_captures(&mut moves);
    if moves.len != 0 {
        return None;
    }
    Some(child)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        8,
        "ranking_data OUTPUT GAMES PLAY_MS LABEL_MS THREADS FIRST_SEED TEACHER_MODEL"
    );
    let games: usize = args[2].parse().unwrap();
    let play_ms: f64 = args[3].parse().unwrap();
    let label_ms: f64 = args[4].parse().unwrap();
    let threads: usize = args[5].parse().unwrap();
    let first_seed: u64 = args[6].parse().unwrap();
    assert!(
        games > 0
            && threads > 0
            && play_ms.is_finite()
            && play_ms > 0.0
            && label_ms.is_finite()
            && label_ms > 0.0
    );
    let model = ValueNet::load(&args[7], 0.5).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])
        .unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(b"CDCRNK01").unwrap();
    let output = Mutex::new(writer);
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let pairs = AtomicUsize::new(0);
    let params = Params::default();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (model, params) = (model.clone(), params.clone());
            let (output, next, completed, pairs) = (&output, &next, &completed, &pairs);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= games {
                    break;
                }
                let seed = first_seed + index as u64;
                let mut rng = Rng::new(seed);
                let mut game = Game::new(&mut rng);
                let mut actor = Searcher::new(
                    params.clone(),
                    SearchConfig {
                        time_ms: play_ms,
                        ..Default::default()
                    },
                    17,
                );
                actor.set_evaluator(Some(LearnedEval::Residual(model.clone())));
                let mut teacher = Searcher::new(
                    params.clone(),
                    SearchConfig {
                        time_ms: label_ms,
                        root_margin: 600,
                        flip_min_depth: 5,
                        ..Default::default()
                    },
                    18,
                );
                teacher.set_evaluator(Some(LearnedEval::Residual(model.clone())));
                let mut records = Vec::new();
                let mut count = 0;
                while game.outcome() == Outcome::Ongoing && game.moves.len() < 600 {
                    if game.pos.assigned
                        && game.moves.len() % 6 == index % 6
                        && game.pos.alive_color(0) + game.pos.alive_color(1) > 5
                    {
                        let label = teacher.think(&game.pos, &game.history);
                        if label.depth >= 3 {
                            let candidates: Vec<_> = label
                                .root
                                .iter()
                                .filter(|r| r.exact && r.score.abs() < 5000)
                                .filter_map(|r| {
                                    quiet_successor(&game.pos, r.mv, &game.history)
                                        .map(|child| (r, child))
                                })
                                .collect();
                            if let Some((best, best_child)) = candidates.first() {
                                let best_features = features(best_child);
                                for (other, child) in candidates
                                    .iter()
                                    .skip(1)
                                    .filter(|(r, _)| (10..=600).contains(&(best.score - r.score)))
                                    .take(3)
                                {
                                    records.extend(seed.to_le_bytes());
                                    records.extend((game.moves.len() as u16).to_le_bytes());
                                    records.extend((label.depth as u16).to_le_bytes());
                                    for mv in [best.mv, other.mv] {
                                        records.extend(mv.encode().to_le_bytes());
                                    }
                                    for score in [best.score, other.score] {
                                        records.extend(score.to_le_bytes());
                                    }
                                    for position in [best_child, child] {
                                        records.extend(evaluate(position, &params).to_le_bytes());
                                    }
                                    for x in [best_features, features(child)] {
                                        for v in x {
                                            records.extend(v.to_le_bytes());
                                        }
                                    }
                                    count += 1;
                                }
                            }
                        }
                    }
                    let mv = if game.pos.assigned {
                        actor.think(&game.pos, &game.history).best
                    } else {
                        Move::flip(rng.below(32) as u8)
                    };
                    assert!(game.legal().contains(&mv));
                    game.play(mv);
                }
                assert_eq!(records.len(), count * (32 + 4 * FEATURES));
                if game.outcome() != Outcome::Ongoing {
                    output.lock().unwrap().write_all(&records).unwrap();
                    pairs.fetch_add(count, Ordering::Relaxed);
                }
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 16 == 0 {
                    eprintln!(
                        "completed {done}/{games}: {} preference pairs",
                        pairs.load(Ordering::Relaxed)
                    );
                }
            });
        }
    });
    output.lock().unwrap().flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_filter_accepts_positions_and_excludes_captures_and_flips() {
        let mut accepted = 0;
        for seed in 0..32 {
            let mut rng = Rng::new(seed);
            let mut game = Game::new(&mut rng);
            for _ in 0..200 {
                if game.outcome() != Outcome::Ongoing {
                    break;
                }
                let legal = game.legal();
                for &mv in &legal {
                    if let Some(mut child) = quiet_successor(&game.pos, mv, &game.history) {
                        assert!(!mv.is_flip());
                        assert!(child.has_moves());
                        assert!(quiet_successor(&game.pos, mv, &[child.hash]).is_none());
                        for _ in 0..2 {
                            let mut captures = MoveList::new();
                            child.gen_captures(&mut captures);
                            assert_eq!(captures.len, 0);
                            child.side ^= 1;
                        }
                        accepted += 1;
                    }
                }
                game.play(legal[rng.below(legal.len() as u64) as usize]);
            }
        }
        assert!(accepted > 100);
    }
}
