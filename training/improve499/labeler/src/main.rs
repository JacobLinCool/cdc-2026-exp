//! Re-label saved public roots using a frozen evaluator and a fixed node budget.
use cdc_engine::{
    board::*,
    eval::{evaluate, Params},
    expanded::ExpandedEval,
    search::{LearnedEval, SearchConfig, Searcher},
    structured::{features, FEATURES},
};
use std::{
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

const BYTES: usize = 1672;

fn decode(record: &[u8]) -> (Pos, Vec<u64>) {
    assert_eq!(record.len(), BYTES);
    let mut pos = Pos {
        cells: record[52..84].try_into().unwrap(),
        pool: record[84..98].try_into().unwrap(),
        alive: record[98..112].try_into().unwrap(),
        side: record[112],
        hidden_n: record[113],
        nocap: u16::from_le_bytes(record[114..116].try_into().unwrap()),
        assigned: true,
        hash: 0,
    };
    assert!(pos.side < 2 && pos.pool.iter().sum::<u8>() == pos.hidden_n);
    assert!(pos
        .pool
        .iter()
        .zip(pos.alive)
        .all(|(&pool, alive)| pool <= alive));
    pos.hash = pos.compute_hash();
    let x: [i16; FEATURES] = std::array::from_fn(|i| {
        i16::from_le_bytes(record[598 + 2 * i..600 + 2 * i].try_into().unwrap())
    });
    assert_eq!(features(&pos), x, "public state/feature mismatch");
    assert_eq!(
        evaluate(&pos, &Params::default()),
        i32::from_le_bytes(record[48..52].try_into().unwrap())
    );
    let n = u16::from_le_bytes(record[116..118].try_into().unwrap()) as usize;
    assert!(n <= 60);
    let history = (0..n)
        .map(|i| u64::from_le_bytes(record[118 + 8 * i..126 + 8 * i].try_into().unwrap()))
        .collect();
    (pos, history)
}

fn quiet_child(pos: &Pos, mv: Move, history: &[u64]) -> Option<Pos> {
    if mv.is_flip() {
        return None;
    }
    let child = pos.make_move(mv);
    if child.nocap >= 50
        || history.contains(&child.hash)
        || child.alive_color(0) == 0
        || child.alive_color(1) == 0
        || !child.has_moves()
    {
        return None;
    }
    let mut captures = MoveList::new();
    child.gen_captures(&mut captures);
    let mut other = child.clone();
    other.side ^= 1;
    other.gen_captures(&mut captures);
    (captures.len == 0).then_some(child)
}

fn write_new(path: &str, header: &[u8], records: impl Iterator<Item = Vec<u8>>) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    let mut out = std::io::BufWriter::new(file);
    out.write_all(header).unwrap();
    for record in records {
        out.write_all(&record).unwrap();
    }
    out.flush().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        7,
        "labeler INPUT ROOT_LABELS PAIRS MODEL NODES THREADS"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    assert!(bytes.len() > 8 && &bytes[..8] == b"CDCSCL01" && (bytes.len() - 8) % BYTES == 0);
    let records: Vec<_> = bytes[8..].chunks_exact(BYTES).collect();
    let model = ExpandedEval::load(&args[4]).unwrap();
    let nodes: u64 = args[5].parse().unwrap();
    let workers: usize = args[6].parse().unwrap();
    assert!(nodes > 0 && workers > 0);
    assert!(!std::path::Path::new(&args[2]).exists() && !std::path::Path::new(&args[3]).exists());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<(Vec<u8>, Vec<u8>)>>> = Mutex::new(vec![None; records.len()]);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let (records, model, next, done, results) =
                (&records, model.clone(), &next, &done, &results);
            scope.spawn(move || {
                let mut searcher = Searcher::new(
                    Params::default(),
                    SearchConfig {
                        time_ms: 1000.0,
                        node_limit: Some(nodes),
                        root_margin: 600,
                        flip_min_depth: 5,
                        ..Default::default()
                    },
                    18,
                );
                searcher.use_tb = false;
                searcher.set_evaluator(Some(LearnedEval::Expanded(model)));
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= records.len() {
                        break;
                    }
                    let record = records[index];
                    let (pos, history) = decode(record);
                    searcher.clear();
                    let label = searcher.think(&pos, &history);
                    assert!(label.nodes <= nodes);
                    let mut legal = MoveList::new();
                    pos.gen_moves(&mut legal);
                    assert!(legal.as_slice().contains(&label.best));
                    let valid = label.depth >= 3 && label.score.abs() < 5000;
                    let mut root = Vec::new();
                    root.extend((index as u32).to_le_bytes());
                    root.extend(label.score.to_le_bytes());
                    root.extend((label.depth as u16).to_le_bytes());
                    root.extend(label.best.encode().to_le_bytes());
                    root.extend(label.nodes.to_le_bytes());
                    root.extend(label.time_ms.to_le_bytes());
                    root.extend((valid as u32).to_le_bytes());
                    assert_eq!(root.len(), 32);
                    let mut pairs = Vec::new();
                    if valid {
                        let mut candidates: Vec<_> = label
                            .root
                            .iter()
                            .filter(|r| r.exact && r.score.abs() < 5000)
                            .filter_map(|r| {
                                quiet_child(&pos, r.mv, &history).map(|child| (r, child))
                            })
                            .collect();
                        candidates.sort_by(|a, b| b.0.score.cmp(&a.0.score));
                        if let Some((best, child)) = candidates.first() {
                            for (other, other_child) in candidates
                                .iter()
                                .skip(1)
                                .filter(|(r, _)| (10..=600).contains(&(best.score - r.score)))
                                .take(3)
                            {
                                pairs.extend(&record[..10]);
                                pairs.extend((label.depth as u16).to_le_bytes());
                                for mv in [best.mv, other.mv] {
                                    pairs.extend(mv.encode().to_le_bytes());
                                }
                                for score in [best.score, other.score] {
                                    pairs.extend(score.to_le_bytes());
                                }
                                for p in [child, other_child] {
                                    pairs.extend(evaluate(p, &Params::default()).to_le_bytes());
                                }
                                for p in [child, other_child] {
                                    for x in features(p) {
                                        pairs.extend(x.to_le_bytes());
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(pairs.len() % (32 + 4 * FEATURES), 0);
                    results.lock().unwrap()[index] = Some((root, pairs));
                    let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if completed % 256 == 0 || completed == records.len() {
                        eprintln!("completed {completed}/{} roots", records.len());
                    }
                }
            });
        }
    });
    let results: Vec<_> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect();
    write_new(&args[2], b"CDCFT001", results.iter().map(|r| r.0.clone()));
    write_new(&args[3], b"CDCRNK01", results.into_iter().map(|r| r.1));
}
