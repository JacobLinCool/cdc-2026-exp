//! Cross-language export check, including exact reconstruction of each public root.
use cdc_engine::{
    board::Pos,
    eval::{evaluate, Params},
    expanded::ExpandedEval,
    structured::{features, FEATURES},
};
use std::io::{Read, Write};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 4, "scaling_predict MODEL DATA OUTPUT");
    let model = ExpandedEval::load(&a[1]).unwrap();
    let bytes = std::fs::metadata(&a[2]).unwrap().len() as usize;
    assert!(
        bytes >= 8 && (bytes - 8) % 1672 == 0,
        "incomplete scaling corpus"
    );
    let mut input = std::io::BufReader::new(std::fs::File::open(&a[2]).unwrap());
    let mut magic = [0; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"CDCSCL01");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[3])
        .unwrap();
    let mut output = std::io::BufWriter::new(file);
    let mut record = [0u8; 1672];
    for _ in 0..(bytes - 8) / 1672 {
        input.read_exact(&mut record).unwrap();
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
        pos.hash = pos.compute_hash();
        let mut x = [0i16; FEATURES];
        for (value, chunk) in x.iter_mut().zip(record[598..].chunks_exact(2)) {
            *value = i16::from_le_bytes(chunk.try_into().unwrap());
        }
        assert_eq!(
            features(&pos),
            x,
            "public state and training features disagree"
        );
        assert_eq!(
            evaluate(&pos, &Params::default()),
            i32::from_le_bytes(record[48..52].try_into().unwrap())
        );
        let prediction = model.evaluate(&pos);
        assert_eq!(prediction, model.evaluate_features(&x));
        output.write_all(&prediction.to_le_bytes()).unwrap();
    }
    output.flush().unwrap();
}
