//! Public root positions with paired search-budget labels, sampled before labeling.
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

const RECORD_BYTES: usize = 1672;
const SAMPLES_PER_GAME: usize = 24;

fn quiet(pos: &Pos) -> bool {
    let mut moves = MoveList::new();
    pos.gen_captures(&mut moves);
    let mut opposite = pos.clone();
    opposite.side ^= 1;
    opposite.gen_captures(&mut moves);
    moves.len == 0
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(
        a.len(),
        10,
        "scaling_data OUTPUT GAMES PLAY_MS LABEL_MS DEEP_MS DEEP_GAMES THREADS FIRST_SEED TEACHER"
    );
    let games: usize = a[2].parse().unwrap();
    let play_ms: f64 = a[3].parse().unwrap();
    let label_ms: f64 = a[4].parse().unwrap();
    let deep_ms: f64 = a[5].parse().unwrap();
    let deep_games: usize = a[6].parse().unwrap();
    let threads: usize = a[7].parse().unwrap();
    let first: u64 = a[8].parse().unwrap();
    assert!(games > 0 && threads > 0 && deep_games <= games);
    assert!([play_ms, label_ms, deep_ms]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0));
    let model = ValueNet::load(&a[9], 0.5).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[1])
        .unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(b"CDCSCL01").unwrap();
    let output = Mutex::new(writer);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let rows = AtomicUsize::new(0);
    let discarded = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let model = model.clone();
            let (output, next, done, rows, discarded) = (&output, &next, &done, &rows, &discarded);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= games {
                    break;
                }
                let seed = first + index as u64;
                let mut rng = Rng::new(seed);
                let mut sampling = Rng::new(seed ^ 0x5343_414c_494e_4701);
                let mut game = Game::new(&mut rng);
                let params = Params::default();
                let mut actor = Searcher::new(
                    params.clone(),
                    SearchConfig {
                        time_ms: play_ms,
                        ..Default::default()
                    },
                    17,
                );
                actor.set_evaluator(Some(LearnedEval::Residual(model.clone())));
                let mut seen = 0;
                let mut positions = Vec::new();
                while game.outcome() == Outcome::Ongoing && game.moves.len() < 600 {
                    if game.pos.assigned
                        && game.moves.len() % 6 == index % 6
                        && game.pos.alive_color(0) + game.pos.alive_color(1) > 5
                        && game.pos.nocap < 50
                        && game.repetition_count() == 1
                    {
                        let start = game.history.len().saturating_sub(60);
                        let item = (
                            game.moves.len() as u16,
                            game.pos.clone(),
                            game.history[start..].to_vec(),
                        );
                        seen += 1;
                        if positions.len() < SAMPLES_PER_GAME {
                            positions.push(item);
                        } else {
                            let replace = sampling.below(seen) as usize;
                            if replace < SAMPLES_PER_GAME {
                                positions[replace] = item;
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
                let mut records = Vec::new();
                if game.outcome() != Outcome::Ongoing {
                    positions.sort_by_key(|p| p.0);
                    let cfg = SearchConfig {
                        time_ms: label_ms,
                        root_margin: 600,
                        flip_min_depth: 5,
                        ..Default::default()
                    };
                    let mut low = Searcher::new(params.clone(), cfg.clone(), 18);
                    let mut high = Searcher::new(
                        params.clone(),
                        SearchConfig {
                            time_ms: deep_ms,
                            ..cfg
                        },
                        18,
                    );
                    low.set_evaluator(Some(LearnedEval::Residual(model.clone())));
                    high.set_evaluator(Some(LearnedEval::Residual(model.clone())));
                    low.use_tb = false;
                    high.use_tb = false;
                    for (ply, pos, history) in positions {
                        low.clear();
                        let label = low.think(&pos, &history);
                        if label.depth < 3 || label.score.abs() >= 5000 {
                            continue;
                        }
                        let is_quiet = quiet(&pos);
                        let deep = if index < deep_games && is_quiet {
                            high.clear();
                            Some(high.think(&pos, &history))
                        } else {
                            None
                        };
                        let valid_deep = deep
                            .as_ref()
                            .is_some_and(|r| r.depth >= 3 && r.score.abs() < 5000);
                        let flags = is_quiet as u16
                            | ((label.best.is_flip() as u16) << 1)
                            | ((deep.as_ref().is_some_and(|r| r.best.is_flip()) as u16) << 2)
                            | ((valid_deep as u16) << 3);
                        let begin = records.len();
                        records.extend(seed.to_le_bytes());
                        records.extend(ply.to_le_bytes());
                        records.extend(flags.to_le_bytes());
                        records.extend((label.depth as u16).to_le_bytes());
                        records.extend((deep.as_ref().map_or(0, |r| r.depth) as u16).to_le_bytes());
                        records.extend(label.score.to_le_bytes());
                        records.extend(deep.as_ref().map_or(0, |r| r.score).to_le_bytes());
                        records.extend((label.time_ms as f32).to_le_bytes());
                        records.extend(
                            (deep.as_ref().map_or(0.0, |r| r.time_ms) as f32).to_le_bytes(),
                        );
                        records.extend(label.nodes.to_le_bytes());
                        records.extend(deep.as_ref().map_or(0, |r| r.nodes).to_le_bytes());
                        records.extend(evaluate(&pos, &params).to_le_bytes());
                        records.extend(pos.cells);
                        records.extend(pos.pool);
                        records.extend(pos.alive);
                        records.push(pos.side);
                        records.push(pos.hidden_n);
                        records.extend(pos.nocap.to_le_bytes());
                        records.extend((history.len() as u16).to_le_bytes());
                        for i in 0..60 {
                            records.extend(history.get(i).copied().unwrap_or(0).to_le_bytes());
                        }
                        let x = features(&pos);
                        assert_eq!(x.len(), FEATURES);
                        for value in x {
                            records.extend(value.to_le_bytes());
                        }
                        assert_eq!(records.len() - begin, RECORD_BYTES);
                    }
                    let mut writer = output.lock().unwrap();
                    writer.write_all(&records).unwrap();
                    writer.flush().unwrap();
                    rows.fetch_add(records.len() / RECORD_BYTES, Ordering::Relaxed);
                } else {
                    discarded.fetch_add(1, Ordering::Relaxed);
                }
                let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 8 == 0 || completed == games {
                    eprintln!(
                        "completed={completed}/{games} rows={} discarded={}",
                        rows.load(Ordering::Relaxed),
                        discarded.load(Ordering::Relaxed)
                    );
                }
            });
        }
    });
}
