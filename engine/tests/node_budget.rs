use cdc_engine::{eval::Params, game::*, search::*};

#[test]
fn node_budgets_are_bounded_repeatable_and_independent_of_time() {
    let mut checked = 0;
    for seed in 0..8 {
        let mut rng = Rng::new(seed + 43000);
        let mut game = Game::new(&mut rng);
        for ply in 0..80 {
            if game.outcome() != Outcome::Ongoing {
                break;
            }
            if ply % 16 == 8 {
                let mut results = Vec::new();
                for time_ms in [0.001, 1e8] {
                    let mut searcher = Searcher::new(
                        Params::default(),
                        SearchConfig {
                            node_limit: Some(2048),
                            time_ms,
                            flip_min_depth: 5,
                            ..Default::default()
                        },
                        12,
                    );
                    searcher.use_tb = false;
                    let result = searcher.think(&game.pos, &game.history);
                    assert!(
                        result.nodes <= 2048,
                        "nodes={} seed={seed} ply={ply}",
                        result.nodes
                    );
                    assert!(game.legal().contains(&result.best));
                    results.push((result.best, result.score, result.nodes, result.depth));
                }
                assert_eq!(results[0], results[1]);
                checked += 1;
            }
            let legal = game.legal();
            game.play(legal[rng.below(legal.len() as u64) as usize]);
        }
    }
    assert!(checked > 20);
}
