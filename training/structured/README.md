# Exact-initialized trainable evaluator

This experiment reconstructs the default engine's numeric evaluation as a trainable computation graph. Its starting coefficients reproduce the original integer scores exactly, including phase blending, piece-value exponentials, rounding and clipping. It retains the existing search and tablebases. The default engine still uses the original evaluator.

A [follow-up experiment](../ranking/README.md) tests stronger-teacher preferences and direct board inference. The first-run results and hashes below are historical evidence from the earlier inference implementation.

The approach is inspired by [handwired's manually constructed numeric circuits](https://github.com/chenglou/handwired/blob/main/docs/README.md). This implementation uses a small NumPy reverse-mode graph and a matching Rust evaluator; it does not depend on handwired or automatically differentiate Rust code. As in [handwired's execution model](https://github.com/chenglou/handwired/blob/main/docs/execution-model.md), exact forward behavior and useful derivatives are separate requirements.

## What is trainable

The model has 169 stored float64 coefficients in a 1,360-byte file. The following nested capacities start from exactly the same default evaluation:

| Capacity | Trainable coefficients | Added freedom |
| --- | ---: | --- |
| `base` | 50 | Active evaluation coefficients in the two existing phases |
| `split` | 123 | 56 zero-initialized per-piece corrections for mobility, protection, trapping and hidden neighbors, plus 12 distance and 5 confinement weights initialized from the original tables |
| `interactions` | 139 | 16 zero-initialized piece/phase, piece/no-progress, cannon/pawn and revealed-count interaction coefficients |

The remaining coefficients retain the default `Params` layout and are fixed. Search-only flip-ordering parameters and the unused linear material mode are not learned. Two distance bins have no support in ordinary opposing-piece pairs; declaring a coefficient trainable does not guarantee a nonzero gradient in every dataset.

Rules and public-board feature extraction stay discrete. Backpropagation differentiates numeric weights, exponentials, products, sums and phase mixtures. The forward pass preserves truncation toward zero at the original locations. The backward pass treats truncation as identity, an explicit straight-through estimator (STE). This is a surrogate gradient, not the mathematical derivative of integer arithmetic. Min/max use the active branch's derivative. There is no gradient through legal moves, alpha-beta cutoffs, action selection or search.

New terms start at zero; the original distance and confinement tables are copied exactly. No random correction changes the initial evaluator. Initial equality is an evaluation and fixed-depth-search property: under a wall-clock limit, inference overhead can still change how far the engine searches.

## Training and verification

512 baseline self-play games at 40 ms/move produced 35,899 sampled positions. Seeds are 6,000,000–6,000,511. A fixed game-level split holds out 102 whole games (7,021 positions) and trains on 410 games (28,878 positions). All capacities use the same split and 60 Adam epochs, choosing the lowest validation loss, including epoch zero.

The target is `sigmoid(search_score / 600)` from the original engine. This first experiment distills a larger search budget; it does not use final outcomes in the loss or learn through self-play policy gradients. Updates are bounded and regularized around the expert initialization. It needs Python + NumPy, with no accelerator or new native dependencies.

| Capacity | Initial validation cross-entropy | Best validation cross-entropy | Best epoch |
| --- | ---: | ---: | ---: |
| `base` | 0.5178217 | 0.5160792 | 30 |
| `split` | 0.5178217 | 0.5145226 | 8 |
| `interactions` | 0.5178217 | 0.5144592 | 30 |

Verification in [verification.json](verification.json):

- All 35,899 initial predictions exactly equal the original evaluator's exported integer scores.
- Rust and NumPy predictions agree exactly on all positions for the initial model and all three trained exports.
- Central finite differences check all 139 declared trainable coordinates of the **unquantized** graph; maximum relative error is 6.47e-8. A separate hand-computed test checks the STE rule. Finite differences are not claimed to validate the surrogate as the integer function's true derivative.
- Rust tests cover more than 10,000 additional positions from legal trajectories, fixed-depth score/move/node equality, hidden-layout isolation, color/reflection invariance and invalid models.
- Repeating the selected training run with the final Python source reproduces its model bytes exactly.
- All 17 Rust tests and the WebAssembly library check pass.

## Match evaluation

Three development runs each used 160 games, 80 paired deals, exchanged seating orders and 25 ms/move. Seeds 7,000,000–7,000,079 are disjoint from training and validation. These runs select a candidate and do not prove superiority.

| Capacity | Wins / draws / losses | Score | Deal-paired 95% interval |
| --- | --- | ---: | --- |
| [`base`](dev-base.json) | 52 / 37 / 71 | 44.06% | 38.44%–49.69% |
| [`split`](dev-split.json) | 61 / 35 / 64 | 49.06% | 43.13%–55.00% |
| `interactions` | 65 / 45 / 50 | 54.69% | 49.38%–60.00% |

`interactions.cse` was selected. Phase mixtures were cached after development to avoid rebuilding invariant parameters during inference; all predictions were reverified before freezing the final executable. The [frozen candidate](frozen-candidate.json) records model, executable and source hashes before holdout play.

The final protocol uses 800 games against the original engine and 800 against the previous residual network (`strong-teacher.nn`, scale 0.5), each on seeds 8,000,000–8,000,399 with exchanged seats at 25 ms/move. The same held-out deals are used for both comparisons. Both runs finish their predetermined size; no optional stopping or retraining on results. Superiority to an opponent requires its paired bootstrap 95% score interval to lie entirely above 50%.

Both predetermined runs completed:

| Opponent | Wins / draws / losses | Score | Deal-paired 95% interval | Relative Elo |
| --- | --- | ---: | --- | ---: |
| [Original engine](holdout-baseline.json) | 272 / 217 / 311 | **47.56%** | 44.94%–50.13% | −17.0 |
| [Previous residual model](holdout-previous.json) | 242 / 225 / 333 | **44.31%** | 41.88%–46.75% | −39.7 |

This candidate **did not improve on the original engine** in the frozen superiority test; its interval includes 50%, so inferiority to that baseline is also not established at this confidence level. It is significantly weaker than the previous residual model under this test. The 54.69% development score did not generalize. The original evaluator remains the default, and the earlier residual network remains the stronger learned candidate.

Actual thinking time was 21.755 / 21.899 ms per move (structured / original) and 21.845 / 21.932 ms (structured / previous). Raw evidence: [baseline games](holdout-baseline.csv), [previous-model games](holdout-previous.csv), and [combined result](result.json). Frozen executable, source and model hashes were rechecked after both runs.

The file size excludes the existing engine and tablebases. Prepared inference parameters occupy approximately 6.4 KB per structured evaluator. A [local microbenchmark](throughput.json) on 1,881 legal-play positions measured approximately 1.51 microseconds for the selected structured evaluator, 1.46 for the previous residual model and 0.90 for the original evaluator. These are machine-dependent microbenchmarks, not search-speed or Elo estimates. Arena thinking-time records include inference and tablebase costs; the engine uses soft time limits.

## Reproduce

From the repository root:

```sh
cargo build --release --manifest-path engine/Cargo.toml \
  --bin structured_data --bin structured_predict --bin arena --bin eval_bench

# Reproduce training using the retained corpus, without overwriting it.
for capacity in base split interactions; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 training/structured/train.py \
    training/structured/selfplay.bin training/structured/selfplay.initial.cse \
    training/structured/retrained-$capacity.cse --capacity "$capacity" --epochs 60
done

python3 training/structured/check.py \
  training/structured/selfplay.bin training/structured/selfplay.initial.cse \
  --predictor engine/target/release/structured_predict \
  --candidate training/structured/base.cse \
  --candidate training/structured/split.cse \
  --candidate training/structured/interactions.cse
```

The local ignored `selfplay.bin` is identified by SHA-256 in [run.json](run.json) and each model's metadata. To generate a separate new corpus, run `engine/target/release/structured_data /tmp/cdc-new-selfplay.bin 512 40 8 6000000` and train against that file and its adjacent `/tmp/cdc-new-selfplay.initial.cse`. Regeneration uses wall-clock search and concurrent game production, so labels and row ordering are not bit-reproducible.

Use fresh seeds for additional experiments; the final holdout is already spent:

```sh
SEED=9000000 ARENA_CSV=training/structured/new-match.csv \
  engine/target/release/arena 100 25 \
  'structured=training/structured/interactions.cse' '' 6

# Replace the empty B specification above to compare with the previous model:
# 'model=training/strong-teacher.nn,model_scale=0.5'

python3 training/report.py training/structured/new-match.csv \
  --model training/structured/interactions.cse --purpose development --time-ms 25
# For the previous-model opponent also add:
# --opponent-model training/strong-teacher.nn --opponent-scale 0.5

engine/target/release/eval_bench \
  training/structured/interactions.cse training/strong-teacher.nn
```

Programmatic loading uses `Searcher::set_evaluator(Some(LearnedEval::Structured(model)))`. `structured=` and `model=` are mutually exclusive; `model_scale` applies only to the residual network.

## Formats

`CDCSE001` is eight magic/version bytes followed by 169 little-endian float64 coefficients: 40 middle-game parameters, 40 endgame parameters, 56 split corrections, 12 distance coefficients, five confinement coefficients and 16 interactions. Unsupported versions, wrong sizes, nonfinite/oversized coefficients, material-mode changes and invalid exponential ratios are rejected. Phase mixtures are prepared once at load time; no training framework is needed at inference.

`CDCDAT01` data consists of an eight-byte header and 1,092-byte records: little-endian seed (`u64`), baseline/search score (`i32` each), outcome (`i16`: loss 0, draw 1, win 2), and 537 public features (`i16` each). The feature layout constants are in `engine/src/structured.rs` and `model.py`. Records contain no secret deal. Unfinished games at the collector's 600-ply bound are discarded; all 512 games were retained in this run. Arena matches use actual rule termination.

## What this establishes

Exact initialization and subsequent gradient-based training work for this evaluator, but this first trained candidate does not beat either benchmark. Lower teacher-score error did not translate into better held-out play. Extra capacity, objective quality and search throughput must be evaluated together; this negative result does not establish a ceiling for the architecture.

Natural next experiments are broader public-position coverage, a stronger teacher or iterative self-play, and ranking candidate moves rather than fitting only scalar scores. Differentiating a limited fixed search tree could follow, but soft choices or cutoff approximations would introduce additional surrogate assumptions. These ideas have not been tested in this experiment.
