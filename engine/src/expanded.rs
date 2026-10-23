//! Function-preserving capacity growth through sparse public-composition tables.
use crate::{
    board::*,
    eval::EVAL_BOUND,
    structured::{StructuredEval, ALIVE, ASSIGNED, FEATURES, HIDDEN_COUNT, POOL},
};
use std::io;

pub const SMALL: usize = 8 * 45;
pub const LARGE: usize = 9 * 7 * 23;
const PREDATOR_OFFSET: [usize; 7] = [0, 6, 8, 12, 18, 26, 36];
const COUNT_OFFSET: [usize; 7] = [0, 2, 5, 8, 11, 14, 17];

#[derive(Clone)]
pub struct ExpandedEval {
    pub base: StructuredEval,
    tier: u32,
    coefficients: Vec<i64>,
}

impl ExpandedEval {
    pub fn load(path: &str) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid CDCEX001 evaluator");
        if bytes.len() < 1372 || &bytes[..8] != b"CDCEX001" {
            return Err(invalid());
        }
        let tier = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let count = match tier {
            0 => 0,
            1 => SMALL,
            2 => SMALL + LARGE,
            _ => return Err(invalid()),
        };
        if bytes.len() != 1372 + count * 8 {
            return Err(invalid());
        }
        let base = StructuredEval::from_bytes(&bytes[12..1372])?;
        let mut coefficients = Vec::with_capacity(count);
        for chunk in bytes[1372..].chunks_exact(8) {
            let value = f64::from_le_bytes(chunk.try_into().unwrap());
            if !value.is_finite() || value.abs() > 4096.0 {
                return Err(invalid());
            }
            coefficients.push((value * 256.0).trunc() as i64);
        }
        Ok(Self {
            base,
            tier,
            coefficients,
        })
    }

    fn correction(&self, alive: &[u8; 14], pool: &[u8; 14], side: usize, hidden: u8) -> i32 {
        if self.tier == 0 {
            return 0;
        }
        let phase = match hidden {
            0 => 0,
            1..=2 => 1,
            3..=5 => 2,
            6..=8 => 3,
            9..=12 => 4,
            13..=18 => 5,
            19..=25 => 6,
            _ => 7,
        };
        let detailed_phase = match hidden {
            0 => 0,
            1..=4 => 1,
            5..=10 => 2,
            11..=20 => 3,
            _ => 4,
        };
        let mut total = 0i64;
        for c in 0..2 {
            let sign = if c == side { 1 } else { -1 };
            let enemy = 1 - c;
            for t in 0..7 {
                let count = alive[c * 7 + t] as i64;
                if count == 0 {
                    continue;
                }
                let predators: usize = (0..7)
                    .filter(|&u| u != t && u != CANNON as usize && ADJ_CAP[u][t])
                    .map(|u| alive[enemy * 7 + u] as usize)
                    .sum();
                let concealed = pool[c * 7 + t] as i64;
                total += sign
                    * count
                    * 7
                    * self.coefficients[phase * 45 + PREDATOR_OFFSET[t] + predators];
                if self.tier == 2 {
                    for (visibility, n) in [(0, count - concealed), (1, concealed)] {
                        if n == 0 {
                            continue;
                        }
                        // No hidden-piece table exists in the fully revealed phase.
                        let group = if detailed_phase == 0 {
                            0
                        } else {
                            detailed_phase * 2 + visibility - 1
                        };
                        let offset = SMALL + (group * 7 + t) * 23;
                        for u in 0..7 {
                            total += sign
                                * n
                                * self.coefficients
                                    [offset + COUNT_OFFSET[u] + alive[enemy * 7 + u] as usize];
                        }
                    }
                }
            }
        }
        // The large table averages over seven opposing piece types. All sums are exact integers.
        (total / (256 * 7)) as i32
    }

    pub fn evaluate(&self, pos: &Pos) -> i32 {
        if !pos.assigned {
            return 0;
        }
        (self.base.evaluate(pos)
            + self.correction(&pos.alive, &pos.pool, pos.side as usize, pos.hidden_n))
        .clamp(-EVAL_BOUND, EVAL_BOUND)
    }

    pub fn evaluate_features(&self, features: &[i16; FEATURES]) -> i32 {
        if features[ASSIGNED] == 0 {
            return 0;
        }
        let alive = std::array::from_fn(|i| features[ALIVE + i] as u8);
        let pool = std::array::from_fn(|i| features[POOL + i] as u8);
        (self.base.evaluate_features(features)
            + self.correction(&alive, &pool, 0, features[HIDDEN_COUNT] as u8))
        .clamp(-EVAL_BOUND, EVAL_BOUND)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::*;

    #[test]
    fn rejects_malformed_expansions() {
        let mut bytes = b"CDCEX001".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(StructuredEval::default().to_bytes());
        bytes.extend(vec![0; SMALL * 8]);
        assert!(ExpandedEval::from_bytes(&bytes).is_ok());
        assert!(ExpandedEval::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(ExpandedEval::from_bytes(&bytes).is_err());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[1372..1380].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(ExpandedEval::from_bytes(&bytes).is_err());
    }

    #[test]
    fn expansion_preserves_initial_scores_and_public_feature_parity() {
        let base = StructuredEval::default();
        let mut checked = 0;
        for tier in 0..=2 {
            let n = match tier {
                0 => 0,
                1 => SMALL,
                _ => SMALL + LARGE,
            };
            let mut bytes = b"CDCEX001".to_vec();
            bytes.extend((tier as u32).to_le_bytes());
            bytes.extend(base.to_bytes());
            bytes.extend(vec![0; n * 8]);
            let initial = ExpandedEval::from_bytes(&bytes).unwrap();
            for (i, chunk) in bytes[1372..].chunks_exact_mut(8).enumerate() {
                chunk.copy_from_slice(&(((i * 29 % 127) as f64 - 63.0) / 9.0).to_le_bytes());
            }
            let perturbed = ExpandedEval::from_bytes(&bytes).unwrap();
            for seed in 0..16 {
                let mut rng = Rng::new(seed);
                let mut game = Game::new(&mut rng);
                for _ in 0..150 {
                    if game.outcome() != Outcome::Ongoing {
                        break;
                    }
                    assert_eq!(initial.evaluate(&game.pos), base.evaluate(&game.pos));
                    let x = crate::structured::features(&game.pos);
                    assert_eq!(
                        perturbed.evaluate(&game.pos),
                        perturbed.evaluate_features(&x)
                    );
                    let mut mirrored = game.pos.clone();
                    mirrored.cells.reverse();
                    assert_eq!(perturbed.evaluate(&game.pos), perturbed.evaluate(&mirrored));
                    let legal = game.legal();
                    game.play(legal[rng.below(legal.len() as u64) as usize]);
                    checked += 1;
                }
            }
        }
        assert!(checked > 5000);
    }
}
