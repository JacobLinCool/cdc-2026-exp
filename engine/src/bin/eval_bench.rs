//! Compare evaluator throughput on the same public positions, with a checksum.
use cdc_engine::{
    board::Pos,
    eval::{evaluate, Params},
    expanded::ExpandedEval,
    game::{Game, Outcome, Rng},
    neural::ValueNet,
    structured::StructuredEval,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 3,
        "eval_bench STRUCTURED_MODEL RESIDUAL_MODEL [EXPANDED_MODELS...]"
    );
    let structured = StructuredEval::load(&args[1]).unwrap();
    let residual = ValueNet::load(&args[2], 0.5).unwrap();
    let params = Params::default();
    let mut positions = Vec::new();
    for seed in 0..64 {
        let mut rng = Rng::new(seed);
        let mut game = Game::new(&mut rng);
        while game.outcome() == Outcome::Ongoing && game.moves.len() < 300 {
            let moves = game.legal();
            game.play(moves[rng.below(moves.len() as u64) as usize]);
            if game.moves.len() % 8 == 0 {
                positions.push(game.pos.clone());
            }
        }
    }
    for position in &positions {
        assert_eq!(
            structured.evaluate(position),
            structured.evaluate_features(&cdc_engine::structured::features(position))
        );
    }
    println!("positions={}", positions.len());
    for kind in ["baseline", "structured", "structured_features", "residual"] {
        measure(kind, &positions, |pos| match kind {
            "baseline" => evaluate(pos, &params),
            "structured" => structured.evaluate(pos),
            "structured_features" => {
                structured.evaluate_features(&cdc_engine::structured::features(pos))
            }
            "residual" => residual.evaluate(pos, evaluate(pos, &params)),
            _ => unreachable!(),
        });
    }
    for path in &args[3..] {
        let expanded = ExpandedEval::load(path).unwrap();
        for pos in &positions {
            assert_eq!(
                expanded.evaluate(pos),
                expanded.evaluate_features(&cdc_engine::structured::features(pos))
            );
        }
        measure(path, &positions, |pos| expanded.evaluate(pos));
    }
}

fn measure(name: &str, positions: &[Pos], evaluator: impl Fn(&Pos) -> i32) {
    let mut timings = Vec::new();
    let mut checksum = 0i64;
    for _ in 0..7 {
        let start = Instant::now();
        let mut sum = 0i64;
        for _ in 0..100 {
            for pos in positions {
                sum += evaluator(black_box(pos)) as i64;
            }
        }
        checksum = black_box(sum);
        timings.push(start.elapsed().as_nanos() as f64 / (100 * positions.len()) as f64);
    }
    timings.sort_by(f64::total_cmp);
    println!("{name}: median_ns={:.1} checksum={checksum}", timings[3]);
}
