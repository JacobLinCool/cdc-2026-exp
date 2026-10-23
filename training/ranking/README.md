# Teacher preferences and faster structured inference

This follow-up tests three changes after the first [structured evaluator experiment](../structured/README.md) failed its held-out comparisons: a teacher using the strongest existing learned evaluator, a loss on candidate-move preferences, and less expensive native inference. The model format stays `CDCSE001` (1,360 bytes, 169 stored coefficients, 139 trainable). The engine search and the default evaluator remain in use as before.

## Inference

Native search now evaluates the board directly with integer sums instead of building and reducing 537 training features at every leaf. Nine phase mixtures, distance/confinement tables and the piece-value exponentials for every possible enemy count are prepared once when loading a model. The feature-based numeric graph remains the reference for data export and training parity; it is not an alternate runtime fallback.

The initial evaluator still matches the original scores exactly. Additional differential tests compare direct inference with the feature graph on legal trajectories under eight perturbed parameter sets. Benchmarks also compare the two paths on the same 1,881 positions and assert equal predictions for the model being measured.

After collection and training stopped, the [same-model benchmark](throughput.json) measured **819 ns** for direct inference versus **1,409 ns** through the feature graph: **41.9% less evaluation time**. Original evaluation measured 834 ns; the previous residual evaluator measured 1,333 ns. These are local microbenchmarks, not whole-search speedups. The serialized model is still 1,360 bytes; prepared parameters occupy approximately 8 KB.


The final selected `value.cse` was also benchmarked after both holdouts: 852 ns direct versus 1426 ns through its feature graph (40.2% less time). The previous residual evaluator measured 1392 ns. See [selected-model timing](throughput-selected.json). Its file is 1,360 bytes versus 4,236 for the previous network; file size excludes prepared caches, engine and tablebases.

## Teacher data

The collector plays 512 games on seeds 10,000,000–10,000,511. The playing policy is the existing residual model (`strong-teacher.nn`, scale 0.5), at 25 ms/move. Every sixth ply, with a game-specific offset, the same model searches at 160 ms with a root margin of 600 and a fixed flip-depth threshold of five. This uses a stronger existing evaluator and a larger labeling budget; it does not establish a new Elo rating for the labeling configuration.

Only root candidates marked **exact** are eligible. Alpha-beta bounds are never treated as exact labels. Eligible candidate successors must:

- come from a legal non-flip move;
- offer neither side an immediate capture;
- have legal replies and both colors alive;
- avoid already-seen successor positions and have fewer than 50 no-progress plies.

The parent must have more than five live pieces, the teacher must complete depth three or more, and scores must be inside ±5,000. Up to three pairs compare the highest-scoring eligible candidate with alternatives 10–600 points below it. These are preferences within the eligible subset, not necessarily the teacher's globally best move. Unfinished games at the collector's 600-ply bound are discarded.

Each pair stores only public successor features, teacher scores, original static scores, move codes, game seed, ply and completed depth. It never records or consults the secret layout for labels. The game simulator uses its deal only when an actual played flip is resolved. The finished corpus contains **5,367 pairs / 10,734 successor records from 451 contributing games** out of 512 played. Games with no eligible pairs contribute no rows. The train/validation split contains 361/90 contributing games and 4,410/957 pairs.

## Objectives

Both learners start from the original engine's parameters and use the same game-level 80/20 split, data, update bounds and 80-epoch budget. The checkpoint is selected by validation loss, with the initial model included as epoch zero.

For candidate A preferred to B, `s_A` and `s_B` are teacher scores from the **parent** player's perspective, while `v_A` and `v_B` evaluate the children from the **opponent**'s perspective:

- The value objective matches `sigmoid(v / 600)` to `sigmoid(-s / 600)`.
- The ranking objective matches `sigmoid((v_B - v_A) / 200)` to `sigmoid((s_A - s_B) / 200)`, plus 0.1 times the value objective.
- Both include a small squared output-drift penalty toward the initial static score and parameter regularization. Normalized parameter updates are bounded to ±1, half the first experiment's bound.

The sign reversal is tested explicitly. Finite differences verify both loss gradients (maximum absolute error 1.87e-13). All 10,734 initial scores match the original engine, and all four exports agree exactly between NumPy and Rust. All 19 Rust tests and the WebAssembly library check pass. See [verification.json](verification.json). Numeric evaluation keeps exact integer forward rounding and the previously verified straight-through surrogate for backward truncation; there is no differentiation through search or board rules.

Strict teacher-preference agreement on the shared validation split is 69.91% initially, 71.26% after value fitting, 71.58% after preference fitting, and 70.43% for the half update. A tie in predicted scores does not count as agreement. These small increases are descriptive validation metrics, not independent evidence of playing strength.

Four development candidates were compared: value fitting, preference fitting, a half-sized preference parameter update, and the previous structured weights with the faster inference code as a control. Parameter interpolation happens offline; it adds no inference branch or runtime scale. Integer truncation and nonlinear terms mean that halving parameter changes does not necessarily halve score changes.

## Evaluation protocol

[experiment-plan.json](experiment-plan.json) records the data, candidates and selection protocol before training completed. Each development candidate plays 160 games against the previous residual champion at 25 ms/move on seeds 11,000,000–11,000,079, exchanging seats on each deal. Highest score wins, with the documented deterministic tie-break. Development performance is selection evidence only.

The selected model and executable were frozen before two fresh 800-game comparisons, against the original engine and the previous residual model, using seeds 12,000,000–12,000,399 at 25 ms/move. Each comparison finished its predetermined sample size. A win claim requires its deal-paired bootstrap 95% score interval entirely above 50%. Earlier holdouts are not reused.

Development results against the previous residual champion:

| Candidate | Wins / draws / losses | Score | Deal-paired 95% interval |
| --- | --- | ---: | --- |
| [value](dev-value.json) | 52 / 47 / 61 | 47.19% | 40.94%–53.44% |
| [rank](dev-rank.json) | 51 / 39 / 70 | 44.06% | 37.81%–50.62% |
| [rank-half](dev-rank-half.json) | 42 / 46 / 72 | 40.62% | 35.31%–45.94% |
| [control](dev-control.json) | 49 / 50 / 61 | 46.25% | 40.31%–52.19% |

The selected candidate is **`value.cse`**, the scalar-fitting objective with the stronger teacher and direct inference. None of the development candidates exceeded 50%. Ranking improved validation preference agreement slightly but did not win the development comparison. Selection still follows the predetermined rule; the [frozen candidate](frozen-candidate.json) records its weights, executable and source hashes. Repeating its training with the retained data reproduced the model bytes exactly.

Both frozen comparisons completed, 800 games / 400 independent paired deals each:

| Opponent | Wins / draws / losses | Score | Deal-paired 95% interval | Relative Elo |
| --- | --- | ---: | --- | ---: |
| [Original engine](holdout-baseline.json) | 307 / 242 / 251 | **53.50%** | 50.94%–56.06% | +24.4 |
| [Previous residual model](holdout-previous.json) | 280 / 253 / 267 | **50.81%** | 48.25%–53.37% | +5.6 |

**The selected model beats the original engine at this tested time control.** Its comparison with the previous residual model is inconclusive: the interval includes 50%, and this was not an equivalence/noninferiority test. The stronger-teacher scalar objective won candidate selection; the preference objective did not. The default evaluator remains unchanged.

Actual thinking time was 21.900 / 21.912 ms per move (candidate / original) and 21.952 / 21.937 ms (candidate / previous model). Both full raw datasets are retained: [original comparison](holdout-baseline.csv), [previous-model comparison](holdout-previous.csv). The [combined report](result.json) verifies model, source and executable hashes after both runs. No model changes or early stopping occurred during holdout play.

## Reproduce

From the repository root, using Rust and Python with NumPy:

Use the selected model on fresh deals (the existing holdout has already been used):

```sh
SEED=13000000 ARENA_CSV=training/ranking/new-match.csv \
  engine/target/release/arena 100 25 \
  'structured=training/ranking/value.cse' \
  'model=training/strong-teacher.nn,model_scale=0.5' 6
```

To reproduce collection and training:

```sh
cargo build --release --manifest-path engine/Cargo.toml \
  --bin ranking_data --bin structured_predict --bin eval_bench --bin arena

# Generate a new corpus without overwriting archived evidence.
engine/target/release/ranking_data /tmp/cdc-new-preferences.bin \
  512 25 160 8 10000000 training/strong-teacher.nn

# Exact training reproduction uses the locally retained, hashed preferences.bin.
for objective in value rank; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 training/ranking/train.py \
    training/ranking/preferences.bin training/structured/selfplay.initial.cse \
    /tmp/cdc-retrained-$objective.cse --objective "$objective" --epochs 80
done
python3 training/ranking/check.py \
  training/ranking/preferences.bin training/structured/selfplay.initial.cse
```

Collection is not bit-reproducible because wall-clock searches and concurrent workers affect labels and record order. Training from the retained corpus uses fixed seeds. Dataset and model hashes, complete game splits and checkpoint losses are recorded with the results.

`CDCRNK01` consists of an eight-byte header followed by 2,180-byte records: seed (`u64`), ply and completed depth (`u16` each), two move codes (`u16`), two teacher scores (`i32`), two original static scores (`i32`), and two 537-element feature arrays (`i16`). All numbers are little-endian. The collector refuses to overwrite an existing data file.

This is an empirical training experiment. Faster exact inference can be established by parity and timing tests; stronger play requires separate matches. Quiet-move preferences cover only part of the game, and lower loss or better preference accuracy alone does not prove an Elo gain.

[Phase diagnostics](phase-diagnostics.json) show that 60.6% of pairs have no hidden pieces, 30.5% have one to seven, and only 9.0% have eight or more. The validation split contains just 89 pairs in that last category. Flips themselves are excluded. This limits coverage of the opening and hidden-information decisions; it is a measured limitation, not a demonstrated explanation for the match outcome.
