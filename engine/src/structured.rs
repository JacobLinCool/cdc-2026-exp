//! Trainable reconstruction of the default evaluator. Quantization truncates toward zero.
//! The public-board feature extractor is shared by training export and native inference.
use crate::{
    board::*,
    eval::{idx, Params, EVAL_BOUND},
};
use std::io;

pub const ALIVE: usize = 0;
pub const POOL: usize = 14;
pub const HUNTED: usize = 28;
pub const CHASE: usize = 42;
pub const DANGER: usize = 210;
pub const MOB: usize = 378;
pub const PROTECT: usize = 392;
pub const TRAPPED: usize = 406;
pub const HIDDEN_NEAR: usize = 420;
pub const CONFINE: usize = 434;
pub const PARITY: usize = 504;
pub const CANNON_LINE: usize = 518;
pub const HIDDEN_COUNT: usize = 532;
pub const NOCAP: usize = 533;
pub const ASSIGNED: usize = 534;
pub const REVEALED: usize = 535;
pub const FEATURES: usize = 537;
pub const SPLIT: usize = 80; // two phases × four terms × seven piece types
pub const NEAR: usize = 136;
pub const CONF: usize = 148;
pub const INTERACTIONS: usize = 153;
pub const PARAMETERS: usize = 169;

#[derive(Clone)]
pub struct StructuredEval {
    weights: [f64; PARAMETERS],
    phases: [Phase; 9],
    near: [i64; 12],
    confinement: [i64; 5],
}

#[derive(Clone, Copy)]
struct Phase {
    weights: [i64; 40],
    split: [[i64; 7]; 4],
    material: [i64; 17],
    kings: [i64; 6],
}

impl Default for StructuredEval {
    fn default() -> Self {
        let p = Params::default();
        let mut weights = [0.0; PARAMETERS];
        for i in 0..40 {
            weights[i] = p.w[i] as f64;
            weights[40 + i] = p.e[i] as f64;
        }
        weights[NEAR..CONF].copy_from_slice(&[0., 64., 40., 26., 17., 11., 7., 4., 2., 1., 0., 0.]);
        weights[CONF..INTERACTIONS].copy_from_slice(&[16., 8., 3., 1., 0.]);
        Self::prepare(weights)
    }
}

impl StructuredEval {
    // Parameters do not change during search. Cache all nine phase mixtures once.
    fn prepare(weights: [f64; PARAMETERS]) -> Self {
        let phases = std::array::from_fn(|hidden| {
            let gate = (8 - hidden) as f64 * 8.0;
            let blend = |a: f64, b: f64| {
                ((a.trunc() * (64.0 - gate) + b.trunc() * gate) / 64.0).trunc() as i64
            };
            let mixed: [i64; 40] = std::array::from_fn(|i| blend(weights[i], weights[40 + i]));
            Phase {
                weights: mixed,
                split: std::array::from_fn(|k| {
                    std::array::from_fn(|t| {
                        blend(weights[SPLIT + k * 7 + t], weights[SPLIT + 28 + k * 7 + t])
                    })
                }),
                material: std::array::from_fn(|n| {
                    (mixed[idx::OBS_TOP] as f64
                        * (mixed[idx::OBS_RATIO] as f64 / 100.0).powi(n as i32))
                    .trunc() as i64
                }),
                kings: std::array::from_fn(|n| {
                    (mixed[idx::OBS_TOP] as f64
                        * (mixed[idx::OBS_KING] as f64 / 100.0).powi(n as i32))
                    .trunc() as i64
                }),
            }
        });
        Self {
            phases,
            near: std::array::from_fn(|i| weights[NEAR + i].trunc() as i64),
            confinement: std::array::from_fn(|i| weights[CONF + i].trunc() as i64),
            weights,
        }
    }

    pub fn load(path: &str) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "invalid CDCSE001 evaluator");
        if bytes.len() != 8 + PARAMETERS * 8 || &bytes[..8] != b"CDCSE001" {
            return Err(bad());
        }
        let mut weights = [0.0; PARAMETERS];
        for (w, b) in weights.iter_mut().zip(bytes[8..].chunks_exact(8)) {
            *w = f64::from_le_bytes(b.try_into().unwrap());
            if !w.is_finite() || w.abs() > 4096.0 {
                return Err(bad());
            }
        }
        for phase in 0..2 {
            if weights[phase * 40 + idx::MAT_MODE] != 1.0 {
                return Err(bad());
            }
            for j in [idx::OBS_RATIO, idx::OBS_KING] {
                if !(5.0..=95.0).contains(&weights[phase * 40 + j]) {
                    return Err(bad());
                }
            }
        }
        Ok(Self::prepare(weights))
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = b"CDCSE001".to_vec();
        for w in self.weights {
            bytes.extend(w.to_le_bytes());
        }
        bytes
    }
    /// Evaluate the public board directly; no dense training histogram is built in search.
    pub fn evaluate(&self, pos: &Pos) -> i32 {
        if !pos.assigned {
            return 0;
        }
        let phase = &self.phases[pos.hidden_n.min(8) as usize];
        let w = &phase.weights;
        let mut values = [[0i64; 7]; 2];
        let mut hunted = [[false; 7]; 2];
        let mut score = [0i64; 2];
        for c in 0..2 {
            let other = 1 - c;
            let mut total = 0;
            for t in 0..7 {
                let mut predators = 0;
                let mut prey = false;
                for u in 0..7 {
                    let count = pos.alive[other * 7 + u] as usize;
                    prey |= count > 0 && ANY_CAP[t][u];
                    if u != CANNON as usize && ADJ_CAP[u][t] {
                        hunted[c][t] |= count > 0;
                        if u != t {
                            predators += count;
                        }
                    }
                }
                let raw = if t == CANNON as usize {
                    w[idx::OBS_CANNON]
                } else if t == KING as usize {
                    phase.kings[pos.alive[other * 7 + PAWN as usize] as usize]
                } else {
                    phase.material[predators]
                };
                let mut value = raw + w[idx::TYPE_ADD + t];
                if !prey {
                    value /= 3;
                }
                values[c][t] = value.max(2);
                let alive = pos.alive[c * 7 + t] as i64;
                let hidden = pos.pool[c * 7 + t] as i64;
                total += alive;
                score[c] += (alive - hidden) * values[c][t]
                    + hidden * values[c][t] * w[idx::HIDDEN_PCT] / 100;
            }
            if total <= 3 {
                score[c] -= w[idx::LAST_PIECES] * (4 - total);
            }
        }
        let mut squares = [[0u8; 16]; 2];
        let mut counts = [0; 2];
        for (s, &piece) in pos.cells.iter().enumerate() {
            if piece < 14 {
                let c = color_of(piece) as usize;
                squares[c][counts[c]] = s as u8;
                counts[c] += 1;
            }
        }
        let hidden = pos.hidden_n as i64;
        let chase_weight = w[idx::CHASE] + w[idx::LATE_CHASE] * (32 - hidden) / 32;
        let attacks = pos.attack_maps();
        for c in 0..2 {
            let other = 1 - c;
            let (mut chase, mut danger, mut confine, mut trapped, mut hidden_near, mut parity) =
                (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
            for &square in &squares[c][..counts[c]] {
                let s = square as usize;
                let t = type_of(pos.cells[s]) as usize;
                let value = values[c][t];
                let mut escape = 0;
                for &neighbor in &NEIGH[s] {
                    if neighbor == 255 {
                        break;
                    }
                    let q = pos.cells[neighbor as usize];
                    if q == EMPTY {
                        score[c] += w[idx::MOBILITY] + phase.split[0][t];
                        if !attacks.attacked(neighbor as usize, t as u8, other) {
                            escape += 1;
                        }
                    } else if q == HIDDEN {
                        hidden_near += w[idx::HIDDEN_NEAR] + phase.split[3][t];
                    } else if color_of(q) as usize == c
                        && type_of(q) as usize <= t
                        && type_of(q) != CANNON
                    {
                        score[c] += w[idx::PROTECT] + phase.split[1][t];
                    }
                }
                if hunted[c][t] {
                    confine += value * self.confinement[escape];
                }
                if escape == 0 && attacks.attacked(s, t as u8, other) {
                    trapped += value * (w[idx::TRAPPED] + phase.split[2][t]);
                }
                for &enemy_square in &squares[other][..counts[other]] {
                    let e = enemy_square as usize;
                    let u = type_of(pos.cells[e]) as usize;
                    let distance = DIST[s][e] as usize;
                    if ADJ_CAP[t][u] && !ADJ_CAP[u][t] {
                        chase += values[other][u] * self.near[distance];
                        if (distance % 2 == 1) == (c == pos.side as usize) {
                            parity += values[other][u];
                        }
                    } else if ADJ_CAP[u][t] && !ADJ_CAP[t][u] {
                        danger += value * self.near[distance];
                    }
                }
                if t == CANNON as usize {
                    for ray in &RAYS[s] {
                        let mut occupied = 0;
                        for &square in ray {
                            if square == 255 {
                                break;
                            }
                            let q = pos.cells[square as usize];
                            if q == EMPTY {
                                continue;
                            }
                            occupied += 1;
                            if occupied >= 2 && q < 14 && color_of(q) as usize == other {
                                score[c] +=
                                    w[idx::CANNON_LINE] * values[other][type_of(q) as usize] / 64;
                                break;
                            }
                            if occupied >= 3 {
                                break;
                            }
                        }
                    }
                }
            }
            score[c] += chase * chase_weight / 4096;
            score[c] -= danger * w[idx::DANGER] / 4096;
            score[c] -= confine * w[idx::CONFINE] * (32 - hidden) / (4096 * 32);
            score[c] -= trapped / 64;
            score[c] += hidden_near * hidden / 32;
            if hidden <= 2 {
                score[c] += parity * w[idx::PARITY] / (64 * (1 + counts[c] as i64));
            }
        }
        let side = pos.side as usize;
        let mut z = [0.0; 16];
        let late = 1.0 - pos.hidden_n as f64 / 32.0;
        let progress = pos.nocap as f64 / 60.0;
        for t in 0..7 {
            let diff = (pos.alive[side * 7 + t] as i16
                - pos.pool[side * 7 + t] as i16
                - pos.alive[(1 - side) * 7 + t] as i16
                + pos.pool[(1 - side) * 7 + t] as i16) as f64;
            z[t] = diff * late;
            z[7 + t] = diff * progress;
        }
        z[14] = (pos.alive[side * 7 + 5] as f64 * pos.alive[(1 - side) * 7 + 6] as f64
            - pos.alive[(1 - side) * 7 + 5] as f64 * pos.alive[side * 7 + 6] as f64)
            / 5.0;
        z[15] = (counts[side] as i16 - counts[1 - side] as i16) as f64 * late * progress;
        let correction = z
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.weights[INTERACTIONS + i])
            .sum::<f64>()
            .trunc() as i64;
        let mut result = score[side] - score[1 - side] + w[idx::TEMPO] + correction;
        if pos.nocap > 10 {
            let keep = w[idx::NOCAP_DECAY].clamp(0, 100);
            let fraction = ((pos.nocap as i64 - 10) * 100 / (DRAW_PLIES as i64 - 10)).min(100);
            let factor = 100 - (100 - keep) * fraction / 100;
            result = result * factor / 100;
        }
        result.clamp(-EVAL_BOUND as i64, EVAL_BOUND as i64) as i32
    }
    pub fn evaluate_features(&self, x: &[i16; FEATURES]) -> i32 {
        if x[ASSIGNED] == 0 {
            return 0;
        }
        let h = x[HIDDEN_COUNT] as f64;
        let phase = &self.phases[(x[HIDDEN_COUNT] as usize).min(8)];
        let w = phase.weights.map(|v| v as f64);
        let split = phase.split.map(|row| row.map(|v| v as f64));
        let mut vals = [[0.0; 7]; 2];
        for (c, row) in vals.iter_mut().enumerate() {
            for (t, v) in row.iter_mut().enumerate() {
                let caps: i16 = (0..7)
                    .filter(|&u| ANY_CAP[t][u])
                    .map(|u| x[ALIVE + (1 - c) * 7 + u])
                    .sum();
                let raw = if t == CANNON as usize {
                    w[idx::OBS_CANNON]
                } else if t == KING as usize {
                    w[idx::OBS_TOP]
                        * (w[idx::OBS_KING] / 100.0)
                            .powi(x[ALIVE + (1 - c) * 7 + PAWN as usize] as i32)
                } else {
                    let n: i16 = (0..7)
                        .filter(|&u| u != CANNON as usize && u != t && ADJ_CAP[u][t])
                        .map(|u| x[ALIVE + (1 - c) * 7 + u])
                        .sum();
                    w[idx::OBS_TOP] * (w[idx::OBS_RATIO] / 100.0).powi(n as i32)
                };
                let mut value = raw.trunc() + w[idx::TYPE_ADD + t];
                if caps == 0 {
                    value = (value / 3.0).trunc();
                }
                *v = value.max(2.0);
            }
        }
        let mut score = [0.0; 2];
        let chase_w = w[idx::CHASE] + (w[idx::LATE_CHASE] * (32.0 - h) / 32.0).trunc();
        for c in 0..2 {
            let mut total = 0.0;
            let mut chase = 0.0;
            let mut danger = 0.0;
            let mut confine = 0.0;
            let mut trapped = 0.0;
            let mut hidden_near = 0.0;
            let mut parity = 0.0;
            for t in 0..7 {
                let a = x[ALIVE + c * 7 + t] as f64;
                let p = x[POOL + c * 7 + t] as f64;
                total += a;
                score[c] +=
                    (a - p) * vals[c][t] + (p * vals[c][t] * w[idx::HIDDEN_PCT] / 100.0).trunc();
                for d in 0..12 {
                    let near = self.near[d] as f64;
                    chase += x[CHASE + (c * 7 + t) * 12 + d] as f64 * vals[1 - c][t] * near;
                    danger += x[DANGER + (c * 7 + t) * 12 + d] as f64 * vals[c][t] * near;
                }
                for e in 0..5 {
                    confine += x[CONFINE + (c * 7 + t) * 5 + e] as f64
                        * vals[c][t]
                        * self.confinement[e] as f64;
                }
                score[c] += x[MOB + c * 7 + t] as f64 * (w[idx::MOBILITY] + split[0][t]);
                score[c] += x[PROTECT + c * 7 + t] as f64 * (w[idx::PROTECT] + split[1][t]);
                trapped +=
                    x[TRAPPED + c * 7 + t] as f64 * vals[c][t] * (w[idx::TRAPPED] + split[2][t]);
                hidden_near +=
                    x[HIDDEN_NEAR + c * 7 + t] as f64 * (w[idx::HIDDEN_NEAR] + split[3][t]);
                parity += x[PARITY + c * 7 + t] as f64 * vals[1 - c][t];
                score[c] += x[CANNON_LINE + c * 7 + t] as f64
                    * (w[idx::CANNON_LINE] * vals[1 - c][t] / 64.0).trunc();
            }
            if total <= 3.0 {
                score[c] -= w[idx::LAST_PIECES] * (4.0 - total);
            }
            score[c] += (chase * chase_w / 4096.0).trunc();
            score[c] -= (danger * w[idx::DANGER] / 4096.0).trunc();
            score[c] -= (confine * w[idx::CONFINE] * (32.0 - h) / (4096.0 * 32.0)).trunc();
            score[c] -= (trapped / 64.0).trunc();
            score[c] += (hidden_near * h / 32.0).trunc();
            if h <= 2.0 {
                score[c] +=
                    (parity * w[idx::PARITY] / (64.0 * (1.0 + x[REVEALED + c] as f64))).trunc();
            }
        }
        let mut s = score[0] - score[1] + w[idx::TEMPO];
        let z = interaction_features(x);
        s += z
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.weights[INTERACTIONS + i])
            .sum::<f64>()
            .trunc();
        if x[NOCAP] > 10 {
            let keep = w[idx::NOCAP_DECAY].clamp(0.0, 100.0);
            let frac = ((x[NOCAP] as f64 - 10.0) * 100.0 / (DRAW_PLIES as f64 - 10.0))
                .trunc()
                .min(100.0);
            let factor = 100.0 - ((100.0 - keep) * frac / 100.0).trunc();
            s = (s * factor / 100.0).trunc();
        }
        s.clamp(-EVAL_BOUND as f64, EVAL_BOUND as f64) as i32
    }
}

pub fn interaction_features(x: &[i16; FEATURES]) -> [f64; 16] {
    let mut z = [0.0; 16];
    let phase = 1.0 - x[HIDDEN_COUNT] as f64 / 32.0;
    let progress = x[NOCAP] as f64 / 60.0;
    for t in 0..7 {
        let diff = (x[ALIVE + t] - x[POOL + t] - x[ALIVE + 7 + t] + x[POOL + 7 + t]) as f64;
        z[t] = diff * phase;
        z[7 + t] = diff * progress;
    }
    z[14] = (x[ALIVE + 5] as f64 * x[ALIVE + 13] as f64
        - x[ALIVE + 12] as f64 * x[ALIVE + 6] as f64)
        / 5.0;
    z[15] = (x[REVEALED] - x[REVEALED + 1]) as f64 * phase * progress;
    z
}

/// All indices are relative to the side to move. No secret deal is accessible.
pub fn features(pos: &Pos) -> [i16; FEATURES] {
    let mut x = [0; FEATURES];
    x[HIDDEN_COUNT] = pos.hidden_n as i16;
    x[NOCAP] = pos.nocap as i16;
    x[ASSIGNED] = pos.assigned as i16;
    let am = pos.attack_maps();
    for c in 0..2 {
        let color = pos.side as usize ^ c;
        let other = color ^ 1;
        for t in 0..7 {
            x[ALIVE + c * 7 + t] = pos.alive[color * 7 + t] as i16;
            x[POOL + c * 7 + t] = pos.pool[color * 7 + t] as i16;
            x[HUNTED + c * 7 + t] = (0..7)
                .any(|u| u != CANNON as usize && ADJ_CAP[u][t] && pos.alive[other * 7 + u] > 0)
                as i16;
        }
        for s in 0..NSQ {
            let q = pos.cells[s];
            if q >= 14 || color_of(q) as usize != color {
                continue;
            }
            x[REVEALED + c] += 1;
            let t = type_of(q) as usize;
            let mut escape = 0;
            for &n in &NEIGH[s] {
                if n == 255 {
                    break;
                }
                let v = pos.cells[n as usize];
                if v == EMPTY {
                    x[MOB + c * 7 + t] += 1;
                    if !am.attacked(n as usize, t as u8, other) {
                        escape += 1;
                    }
                } else if v == HIDDEN {
                    x[HIDDEN_NEAR + c * 7 + t] += 1;
                } else if color_of(v) as usize == color
                    && type_of(v) as usize <= t
                    && type_of(v) != CANNON
                {
                    x[PROTECT + c * 7 + t] += 1;
                }
            }
            if x[HUNTED + c * 7 + t] != 0 {
                x[CONFINE + (c * 7 + t) * 5 + escape] += 1;
            }
            if escape == 0 && am.attacked(s, t as u8, other) {
                x[TRAPPED + c * 7 + t] += 1;
            }
            for e in 0..NSQ {
                let v = pos.cells[e];
                if v >= 14 || color_of(v) as usize != other {
                    continue;
                }
                let u = type_of(v) as usize;
                let d = DIST[s][e] as usize;
                if ADJ_CAP[t][u] && !ADJ_CAP[u][t] {
                    x[CHASE + (c * 7 + u) * 12 + d] += 1;
                    if (d % 2 == 1) == (c == 0) {
                        x[PARITY + c * 7 + u] += 1;
                    }
                } else if ADJ_CAP[u][t] && !ADJ_CAP[t][u] {
                    x[DANGER + (c * 7 + t) * 12 + d] += 1;
                }
            }
            if t == CANNON as usize {
                for ray in &RAYS[s] {
                    let mut count = 0;
                    for &n in ray {
                        if n == 255 {
                            break;
                        }
                        let v = pos.cells[n as usize];
                        if v == EMPTY {
                            continue;
                        }
                        count += 1;
                        if count >= 2 && v < 14 && color_of(v) as usize == other {
                            x[CANNON_LINE + c * 7 + type_of(v) as usize] += 1;
                            break;
                        }
                        if count >= 3 {
                            break;
                        }
                    }
                }
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        eval::evaluate,
        game::{Game, Outcome, Rng},
    };
    #[test]
    fn initial_model_matches_original_over_legal_trajectories() {
        let net = StructuredEval::default();
        let params = Params::default();
        let mut tested = 0;
        for seed in 0..128 {
            let mut rng = Rng::new(seed);
            let mut game = Game::new(&mut rng);
            for _ in 0..300 {
                assert_eq!(
                    net.evaluate(&game.pos),
                    evaluate(&game.pos, &params),
                    "seed {seed} ply {}",
                    game.moves.len()
                );
                tested += 1;
                if game.outcome() != Outcome::Ongoing {
                    break;
                }
                let moves = game.legal();
                game.play(moves[rng.below(moves.len() as u64) as usize]);
            }
        }
        assert!(tested > 10000);
    }
    #[test]
    fn model_roundtrip_and_rejection() {
        let model = StructuredEval::default();
        let mut bytes = model.to_bytes();
        assert_eq!(
            StructuredEval::from_bytes(&bytes).unwrap().weights,
            model.weights
        );
        assert!(StructuredEval::from_bytes(&bytes[..20]).is_err());
        bytes[8..16].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(StructuredEval::from_bytes(&bytes).is_err());
    }

    #[test]
    fn direct_inference_matches_feature_graph_with_perturbed_weights() {
        let mut rng = Rng::new(771);
        let initial = StructuredEval::default();
        let models: Vec<_> = (0..8)
            .map(|_| {
                let mut weights = initial.weights;
                for (i, value) in weights.iter_mut().enumerate() {
                    if i == idx::MAT_MODE || i == 40 + idx::MAT_MODE {
                        continue;
                    }
                    *value += (rng.below(2001) as f64 / 1000.0 - 1.0) * value.abs().max(10.0) * 0.2;
                }
                StructuredEval::from_bytes(&StructuredEval::prepare(weights).to_bytes()).unwrap()
            })
            .collect();
        let mut checked = 0;
        for seed in 0..64 {
            let mut game = Game::new(&mut Rng::new(seed + 800));
            for _ in 0..300 {
                let model = &models[checked % models.len()];
                assert_eq!(
                    model.evaluate(&game.pos),
                    model.evaluate_features(&features(&game.pos)),
                    "seed {seed}, ply {}",
                    game.moves.len()
                );
                checked += 1;
                if game.outcome() != Outcome::Ongoing {
                    break;
                }
                let legal = game.legal();
                game.play(legal[rng.below(legal.len() as u64) as usize]);
            }
        }
        assert!(checked > 5000);
    }
}
