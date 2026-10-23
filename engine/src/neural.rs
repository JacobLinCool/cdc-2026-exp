//! Small residual value network. Inputs contain public information only.
use crate::board::*;
use crate::eval::EVAL_BOUND;
use std::io;

pub const INPUTS: usize = 64;
pub const HIDDEN_UNITS: usize = 16;
pub const WEIGHTS: usize = INPUTS * HIDDEN_UNITS + HIDDEN_UNITS * 2 + 1;

#[derive(Clone)]
pub struct ValueNet {
    weights: [f32; WEIGHTS],
    scale: f32,
}

impl ValueNet {
    pub fn load(path: &str, scale: f32) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, scale)
    }

    pub fn from_bytes(bytes: &[u8], scale: f32) -> io::Result<Self> {
        if bytes.len() != 8 + WEIGHTS * 4
            || &bytes[..8] != b"CDCNN001"
            || !scale.is_finite()
            || !(0.0..=2.0).contains(&scale)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid CDCNN001 model or scale",
            ));
        }
        let mut weights = [0.0; WEIGHTS];
        for (w, b) in weights.iter_mut().zip(bytes[8..].chunks_exact(4)) {
            *w = f32::from_le_bytes(b.try_into().unwrap());
            if !w.is_finite() || w.abs() > 100.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid model weight",
                ));
            }
        }
        Ok(Self { weights, scale })
    }

    pub fn evaluate(&self, pos: &Pos, baseline: i32) -> i32 {
        if !pos.assigned {
            return 0;
        }
        let x = features(pos, baseline);
        (baseline as f32 + self.scale * 600.0 * self.residual(&x))
            .round()
            .clamp(-EVAL_BOUND as f32, EVAL_BOUND as f32) as i32
    }

    fn residual(&self, x: &[f32; INPUTS]) -> f32 {
        let mut y = self.weights[WEIGHTS - 1];
        for h in 0..HIDDEN_UNITS {
            let mut z = self.weights[INPUTS * HIDDEN_UNITS + h];
            for (i, &v) in x.iter().enumerate() {
                z += self.weights[h * INPUTS + i] * v;
            }
            y += z.max(0.0) * self.weights[INPUTS * HIDDEN_UNITS + HIDDEN_UNITS + h];
        }
        y
    }
}

/// Side-relative counts, threats, safe mobility, material and game phase.
/// Never accepts Game (which owns the secret deal).
pub fn features(pos: &Pos, baseline: i32) -> [f32; INPUTS] {
    let mut x = [0.0; INPUTS];
    let am = pos.attack_maps();
    for relative in 0..2usize {
        let c = pos.side as usize ^ relative;
        for t in 0..7 {
            x[relative * 7 + t] = pos.alive[c * 7 + t] as f32 / INIT_COUNT[t] as f32;
            x[14 + relative * 7 + t] = pos.pool[c * 7 + t] as f32 / INIT_COUNT[t] as f32;
        }
        for (s, &q) in pos.cells.iter().enumerate() {
            if q >= 14 || color_of(q) as usize != c {
                continue;
            }
            let t = type_of(q) as usize;
            if am.attacked(s, t as u8, c ^ 1) {
                x[28 + relative * 7 + t] += 1.0 / INIT_COUNT[t] as f32;
            }
            for &n in &NEIGH[s] {
                if n == 255 {
                    break;
                }
                if pos.cells[n as usize] == EMPTY && !am.attacked(n as usize, t as u8, c ^ 1) {
                    x[42 + relative * 7 + t] += 0.25 / INIT_COUNT[t] as f32;
                }
            }
        }
        x[59 + relative] = pos.alive_color(c as u8) as f32 / 16.0;
    }
    x[56] = baseline as f32 / 1000.0;
    x[57] = pos.hidden_n as f32 / 32.0;
    x[58] = pos.nocap as f32 / DRAW_PLIES as f32;
    x[61] = x[56] * x[58];
    x[62] = x[56] * x[57];
    x[63] = (x[59] - x[60]) * (1.0 - x[57]);
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Rng};
    #[test]
    fn hidden_layout_cannot_change_features() {
        let mut a = Game::new(&mut Rng::new(1));
        a.play(Move::flip(0));
        let mut b = a.clone();
        b.layout.reverse();
        assert_eq!(features(&a.pos, 123), features(&b.pos, 123));
    }
    #[test]
    fn zero_residual_preserves_baseline_and_bound() {
        let mut g = Game::new(&mut Rng::new(2));
        g.play(Move::flip(0));
        let net = ValueNet {
            weights: [0.0; WEIGHTS],
            scale: 1.0,
        };
        assert_eq!(net.evaluate(&g.pos, 321), 321);
        assert_eq!(net.evaluate(&g.pos, 20000), EVAL_BOUND);
    }
    #[test]
    fn rejects_corrupt_models() {
        assert!(ValueNet::from_bytes(b"CDCNN001", 1.0).is_err());
        let mut bytes = b"CDCNN001".to_vec();
        bytes.extend(vec![0; WEIGHTS * 4]);
        assert!(ValueNet::from_bytes(&bytes, 1.0).is_ok());
        assert!(ValueNet::from_bytes(&bytes, f32::NAN).is_err());
        bytes[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ValueNet::from_bytes(&bytes, 1.0).is_err());
    }
    #[test]
    fn exported_network_matches_numpy() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../training/");
        let net = ValueNet::load(&format!("{root}value.nn"), 1.0).unwrap();
        let fixture = std::fs::read_to_string(format!("{root}value.fixture.txt")).unwrap();
        for line in fixture.lines() {
            let row: Vec<f32> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let x: [f32; INPUTS] = row[1..65].try_into().unwrap();
            let actual = row[0] + 600.0 * net.residual(&x);
            assert!((actual - row[65]).abs() < 0.01, "{actual} vs {}", row[65]);
        }
    }
    #[test]
    fn features_are_color_and_reflection_invariant() {
        let mut game = Game::new(&mut Rng::new(72));
        let mut rng = Rng::new(96);
        for _ in 0..20 {
            let moves = game.legal();
            game.play(moves[rng.below(moves.len() as u64) as usize]);
        }
        let mut p = game.pos.clone();
        p.side ^= 1;
        for q in &mut p.cells {
            if *q < 14 {
                *q = piece(color_of(*q) ^ 1, type_of(*q));
            }
        }
        p.pool.rotate_left(7);
        p.alive.rotate_left(7);
        assert_eq!(features(&game.pos, 123), features(&p, 123));
        p.cells.reverse();
        assert_eq!(features(&game.pos, 123), features(&p, 123));
    }
}
