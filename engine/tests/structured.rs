use cdc_engine::{board::*, eval::Params, game::*, search::*, structured::*};

#[test]
fn initial_model_preserves_fixed_depth_scores_moves_and_nodes() {
    for seed in 0..4 {
        let mut rng = Rng::new(900 + seed);
        let mut game = Game::new(&mut rng);
        for ply in 0..64 {
            if game.outcome() != Outcome::Ongoing {
                break;
            }
            if [16, 32, 48].contains(&ply) {
                let cfg = SearchConfig {
                    max_depth: 2,
                    time_ms: 1e8,
                    ..Default::default()
                };
                let mut baseline = Searcher::new(Params::default(), cfg.clone(), 12);
                let mut learned = Searcher::new(Params::default(), cfg, 12);
                baseline.use_tb = false;
                learned.use_tb = false;
                learned.set_evaluator(Some(LearnedEval::Structured(StructuredEval::default())));
                let a = baseline.think(&game.pos, &game.history);
                let b = learned.think(&game.pos, &game.history);
                assert_eq!(
                    (a.score, a.best, a.nodes, a.depth),
                    (b.score, b.best, b.nodes, b.depth)
                );
            }
            let legal = game.legal();
            game.play(legal[rng.below(legal.len() as u64) as usize]);
        }
    }
}

#[test]
fn public_features_ignore_secret_deal_and_respect_symmetries() {
    let mut rng = Rng::new(700);
    let mut game = Game::new(&mut rng);
    for _ in 0..32 {
        let legal = game.legal();
        game.play(legal[rng.below(legal.len() as u64) as usize]);
    }
    let original = features(&game.pos);
    game.layout.reverse();
    assert_eq!(features(&game.pos), original);
    game.pos.cells.reverse();
    assert_eq!(features(&game.pos), original);
    game.pos.side ^= 1;
    game.pos.pool.rotate_left(7);
    game.pos.alive.rotate_left(7);
    for cell in &mut game.pos.cells {
        if *cell < 14 {
            *cell = piece(color_of(*cell) ^ 1, type_of(*cell));
        }
    }
    assert_eq!(features(&game.pos), original);
}
