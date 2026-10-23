//! 自我對弈擂台：arena <games> <time_ms> "A 覆寫" "B 覆寫" [threads]
//! 覆寫格式：key=val,key=val（參數名見 eval::PARAM_NAMES，或 fr / iflips / rqflips / lmr / variety / tmul）
use cdc_engine::board::*;
use cdc_engine::eval::*;
use cdc_engine::game::*;
use cdc_engine::moe::*;
use cdc_engine::search::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Spec {
    pub params: Params,
    pub cfg: SearchConfig,
    pub variety: f64,
    pub tmul: f64,
    pub brain: bool,
    pub use_tb: bool,
    pub model: Option<LearnedEval>,
}

pub fn parse_spec(s: &str) -> Spec {
    let mut sp = Spec {
        params: Params::default(),
        cfg: SearchConfig::default(),
        variety: 0.0,
        tmul: 1.0,
        brain: false,
        use_tb: true,
        model: None,
    };
    let mut model_path = None;
    let mut structured_path = None;
    let mut expanded_path = None;
    let mut model_scale = None;
    for kv in s.split(',').filter(|x| !x.is_empty()) {
        let mut it = kv.splitn(2, '=');
        let k = it.next().unwrap().trim();
        if k == "expanded" {
            expanded_path = Some(it.next().expect("expanded model path required").trim());
            continue;
        }
        if k == "structured" {
            structured_path = Some(it.next().expect("structured model path required").trim());
            continue;
        }
        if k == "model" {
            model_path = Some(it.next().expect("model path required").trim());
            continue;
        }
        let v: f64 = it
            .next()
            .expect("expected key=value")
            .trim()
            .parse()
            .expect("bad value");
        match k {
            "nodes" => {
                assert!(v.is_finite() && v >= 1.0 && v <= (1u64 << 53) as f64 && v.fract() == 0.0);
                sp.cfg.node_limit = Some(v as u64);
            }
            "model_scale" => {
                assert!(v.is_finite() && (0.0..=2.0).contains(&v));
                model_scale = Some(v as f32);
            }
            "fr" => sp.cfg.flip_reduction = v as i32,
            "iflips" => sp.cfg.interior_flips = v as usize,
            "rqflips" => sp.cfg.root_quiet_flips = v as usize,
            "lmr" => sp.cfg.use_lmr = v != 0.0,
            "variety" => {
                sp.variety = v;
                sp.brain = true;
            }
            "brain" => sp.brain = v != 0.0,
            "margin" => sp.cfg.root_margin = v as i32,
            "tmul" => sp.tmul = v,
            "depth" => sp.cfg.max_depth = v as i32,
            "contempt" => sp.cfg.contempt = v as i32,
            "nmp" => sp.cfg.null_move = v != 0.0,
            "fmd" => sp.cfg.flip_min_depth = v as i32,
            "tb" => sp.use_tb = v != 0.0,
            "idstop" => sp.cfg.id_stop = v,
            _ => {
                if !sp.params.set(k, v.round() as i32) {
                    panic!("unknown key {k}");
                }
            }
        }
    }
    assert!(
        model_path.is_some() || model_scale.is_none(),
        "model_scale requires a model"
    );
    sp.model = model_path.map(|path| {
        LearnedEval::Residual(
            cdc_engine::neural::ValueNet::load(path, model_scale.unwrap_or(1.0))
                .expect("invalid model"),
        )
    });
    if let Some(path) = structured_path {
        assert!(sp.model.is_none(), "choose exactly one learned evaluator");
        sp.model = Some(LearnedEval::Structured(
            cdc_engine::structured::StructuredEval::load(path).expect("invalid structured model"),
        ));
    }
    if let Some(path) = expanded_path {
        assert!(sp.model.is_none(), "choose exactly one learned evaluator");
        sp.model = Some(LearnedEval::Expanded(
            cdc_engine::expanded::ExpandedEval::load(path).expect("invalid expanded model"),
        ));
    }
    sp
}

pub struct Player {
    brain: Brain,
    use_brain: bool,
    time: f64,
    pub elapsed_ms: f64,
    pub move_count: usize,
    pub nodes: u64,
    pub depth_sum: u64,
}

impl Player {
    pub fn new(sp: &Spec, time: f64, seed: u64, tt_bits: u32) -> Player {
        let mut brain = Brain::new(seed, tt_bits);
        brain.searcher.params = sp.params.clone();
        brain.searcher.set_evaluator(sp.model.clone());
        let margin = brain.searcher.cfg.root_margin;
        brain.searcher.cfg = sp.cfg.clone();
        if sp.brain {
            brain.searcher.cfg.root_margin = margin.max(sp.cfg.root_margin);
        }
        brain.variety = sp.variety;
        brain.searcher.use_tb = sp.use_tb;
        Player {
            brain,
            use_brain: sp.brain,
            time: time * sp.tmul,
            elapsed_ms: 0.0,
            move_count: 0,
            nodes: 0,
            depth_sum: 0,
        }
    }
    pub fn choose(&mut self, g: &Game, rng: &mut Rng) -> Move {
        let start = std::time::Instant::now();
        let mv = self.choose_inner(g, rng);
        self.elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
        self.move_count += 1;
        mv
    }

    fn choose_inner(&mut self, g: &Game, rng: &mut Rng) -> Move {
        if !g.pos.assigned {
            return Move::flip(rng.below(32) as u8);
        }
        if self.use_brain {
            self.brain.decide(g, self.time).mv
        } else {
            self.brain.searcher.cfg.time_ms = self.time;
            let r = self.brain.searcher.think(&g.pos, &g.history);
            if let Some(limit) = self.brain.searcher.cfg.node_limit {
                assert!(r.nodes <= limit, "search exceeded node budget");
            }
            self.nodes += r.nodes;
            self.depth_sum += r.depth as u64;
            if std::env::var("VS").is_ok() {
                eprintln!("  score={} depth={}", r.score, r.depth);
            }
            r.best
        }
    }
}

#[derive(Clone, Copy)]
pub struct GameRecord {
    score: f64,
    plies: usize,
    a_ms: f64,
    b_ms: f64,
    a_moves: usize,
    b_moves: usize,
    a_nodes: u64,
    b_nodes: u64,
    a_depth: u64,
    b_depth: u64,
}

pub fn play_game(
    seed: u64,
    a_first: bool,
    time: f64,
    a: &Spec,
    b: &Spec,
    verbose: bool,
) -> GameRecord {
    let mut rng = Rng::new(seed);
    let mut g = Game::new(&mut rng);
    let mut pa = Player::new(a, time, seed * 3 + 1, 18);
    let mut pb = Player::new(b, time, seed * 5 + 2, 18);
    let mut a_color: Option<u8> = None;
    let mut turn_a = a_first;
    loop {
        let o = g.outcome();
        assert!(
            g.moves.len() <= 4096,
            "game failed to terminate under progress rules"
        );
        if o != Outcome::Ongoing {
            let s = match (o, a_color) {
                (Outcome::Win(c), Some(ac)) => {
                    if c == ac {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.5,
            };
            if verbose {
                eprintln!("result for A: {s}");
            }
            if s == 0.5 && std::env::var("SHOWDRAW").is_ok() {
                let e = evaluate(&g.pos, &Params::default());
                eprintln!(
                    "DRAW nocap={} eval(stm)={} pieces={}/{} \n{}",
                    g.pos.nocap,
                    e,
                    g.pos.alive_color(g.pos.side),
                    g.pos.alive_color(g.pos.side ^ 1),
                    g.pos.to_string_board()
                );
            }
            return GameRecord {
                score: s,
                plies: g.moves.len(),
                a_ms: pa.elapsed_ms,
                b_ms: pb.elapsed_ms,
                a_moves: pa.move_count,
                b_moves: pb.move_count,
                a_nodes: pa.nodes,
                b_nodes: pb.nodes,
                a_depth: pa.depth_sum,
                b_depth: pb.depth_sum,
            };
        }
        let m = if turn_a {
            pa.choose(&g, &mut rng)
        } else {
            pb.choose(&g, &mut rng)
        };
        let first = !g.pos.assigned;
        assert!(g.legal().contains(&m), "illegal arena move");
        let info = g.play(m);
        if first {
            let c = color_of(info);
            a_color = Some(if turn_a { c } else { c ^ 1 });
        }
        if verbose {
            eprintln!(
                "{} {} nocap={}\n{}",
                if turn_a { "A" } else { "B" },
                m,
                g.pos.nocap,
                g.pos.to_string_board()
            );
        }
        turn_a = !turn_a;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() <= 6, "arena GAMES MS A_SPEC B_SPEC THREADS");
    let games: usize = args
        .get(1)
        .map(|s| s.parse().expect("invalid game count"))
        .unwrap_or(20);
    let time: f64 = args
        .get(2)
        .map(|s| s.parse().expect("invalid time budget"))
        .unwrap_or(50.0);
    let a = parse_spec(args.get(3).map(|s| s.as_str()).unwrap_or(""));
    let b = parse_spec(args.get(4).map(|s| s.as_str()).unwrap_or(""));
    let threads: usize = args
        .get(5)
        .map(|s| s.parse().expect("invalid thread count"))
        .unwrap_or(16);
    let seed0: u64 = match std::env::var("SEED") {
        Ok(s) => s.parse().expect("invalid SEED"),
        Err(std::env::VarError::NotPresent) => 1000,
        Err(e) => panic!("invalid SEED: {e}"),
    };
    assert!(
        games >= 4 && games % 2 == 0 && threads > 0 && time.is_finite() && time > 0.0,
        "need an even game count >= 4, positive threads and time"
    );
    let verbose = std::env::var("V").is_ok();
    let records = Arc::new(Mutex::new(vec![None; games]));
    let next = Arc::new(AtomicUsize::new(0));
    let res = Arc::new(Mutex::new((0usize, 0usize, 0usize, 0usize)));
    let mut hs = vec![];
    for _ in 0..threads {
        let (next, res, a, b) = (next.clone(), res.clone(), a.clone(), b.clone());
        let records = records.clone();
        hs.push(std::thread::spawn(move || loop {
            let i = next.fetch_add(1, Ordering::SeqCst);
            if i >= games {
                break;
            }
            // 同一個牌局兩邊各先手一次
            let record = play_game(seed0 + (i / 2) as u64, i % 2 == 0, time, &a, &b, verbose);
            records.lock().unwrap()[i] = Some(record);
            let (s, len) = (record.score, record.plies);
            let mut r = res.lock().unwrap();
            if s == 1.0 {
                r.0 += 1
            } else if s == 0.0 {
                r.2 += 1
            } else {
                r.1 += 1
            }
            r.3 += len;
            let done = r.0 + r.1 + r.2;
            if done % 20 == 0 {
                eprintln!("completed {done}/{games}: {}/{}/{}", r.0, r.1, r.2);
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    let records = records.lock().unwrap();
    if let Ok(path) = std::env::var("ARENA_CSV") {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(out, "seed,a_first,score,plies,a_ms,b_ms,a_moves,b_moves,a_nodes,b_nodes,a_depth,b_depth").unwrap();
        for (i, record) in records.iter().enumerate() {
            let GameRecord {
                score,
                plies,
                a_ms,
                b_ms,
                a_moves,
                b_moves,
                a_nodes,
                b_nodes,
                a_depth,
                b_depth,
            } = record.unwrap();
            writeln!(
                out,
                "{},{},{score},{plies},{a_ms:.3},{b_ms:.3},{a_moves},{b_moves},{a_nodes},{b_nodes},{a_depth},{b_depth}",
                seed0 + (i / 2) as u64,
                i % 2 == 0
            )
            .unwrap();
        }
        out.flush().unwrap();
    }
    let paired: Vec<f64> = records
        .chunks_exact(2)
        .map(|r| (r[0].unwrap().score + r[1].unwrap().score) / 2.0)
        .collect();
    let mean = paired.iter().sum::<f64>() / paired.len() as f64;
    let se = (paired.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
        / ((paired.len() - 1) * paired.len()) as f64)
        .sqrt();
    println!(
        "Deal-paired score={mean:.4}, normal 95% CI=[{:.4}, {:.4}] ({} independent deals)",
        (mean - 1.96 * se).max(0.0),
        (mean + 1.96 * se).min(1.0),
        paired.len()
    );
    let r = res.lock().unwrap();
    let n = (r.0 + r.1 + r.2) as f64;
    let sc = (r.0 as f64 + 0.5 * r.1 as f64) / n;
    let elo = if sc > 0.0 && sc < 1.0 {
        -400.0 * (1.0 / sc - 1.0).log10()
    } else {
        f64::NAN
    };
    println!(
        "A W/D/L = {}/{}/{}  score={:.3} elo={:.0} avglen={:.0}",
        r.0,
        r.1,
        r.2,
        sc,
        elo,
        r.3 as f64 / n
    );
}
