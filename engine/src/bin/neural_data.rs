//! Generate public-position training records from baseline engine self-play.
use cdc_engine::{board::*, eval::*, game::*, neural::features, search::*};
use std::{
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "neural_data OUTPUT GAMES MS THREADS FIRST_SEED"
    );
    let output = Arc::new(Mutex::new(BufWriter::new(
        std::fs::File::create(&args[1]).unwrap(),
    )));
    let games: usize = args[2].parse().unwrap();
    let ms: f64 = args[3].parse().unwrap();
    let threads: usize = args[4].parse().unwrap();
    let seed: u64 = args[5].parse().unwrap();
    assert!(games > 0 && threads > 0 && ms.is_finite() && ms > 0.0);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (output, next) = (output.clone(), &next);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= games {
                    break;
                }
                let mut rng = Rng::new(seed + i as u64);
                let mut game = Game::new(&mut rng);
                let mut search = Searcher::new(
                    Params::default(),
                    SearchConfig {
                        time_ms: ms,
                        ..Default::default()
                    },
                    17,
                );
                let mut records = Vec::new();
                while game.outcome() == Outcome::Ongoing && game.moves.len() < 600 {
                    let mv = if !game.pos.assigned {
                        Move::flip(rng.below(32) as u8)
                    } else {
                        let r = search.think(&game.pos, &game.history);
                        if r.depth > 0 && game.moves.len() % 2 == 0 {
                            let base = evaluate(&game.pos, &search.params);
                            records.push((game.pos.side, base, r.score, features(&game.pos, base)));
                        }
                        r.best
                    };
                    assert!(game.legal().contains(&mv));
                    game.play(mv);
                }
                // Truncated games have no trustworthy outcome label.
                if game.outcome() == Outcome::Ongoing {
                    continue;
                }
                let mut rows = String::new();
                for (side, base, teacher, x) in records {
                    let result = match game.outcome() {
                        Outcome::Win(c) => {
                            if c == side {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => 0.5,
                    };
                    rows.push_str(&format!("{},{base},{teacher},{result}", seed + i as u64));
                    for v in x {
                        rows.push_str(&format!(",{v:.7}"));
                    }
                    rows.push('\n');
                }
                output.lock().unwrap().write_all(rows.as_bytes()).unwrap();
                if (i + 1) % 16 == 0 {
                    eprintln!("generated game {}/{}", i + 1, games);
                }
            });
        }
    });
    output.lock().unwrap().flush().unwrap();
}
