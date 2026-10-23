//! Evaluate exported public-feature records for cross-language numerical parity.
use cdc_engine::structured::{StructuredEval, FEATURES};
use std::io::{Read, Write};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 4, "structured_predict MODEL DATA OUTPUT");
    let model = StructuredEval::load(&a[1]).unwrap();
    let mut input = std::io::BufReader::new(std::fs::File::open(&a[2]).unwrap());
    let mut magic = [0; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"CDCDAT01");
    let mut output = std::io::BufWriter::new(std::fs::File::create(&a[3]).unwrap());
    let size = 18 + FEATURES * 2;
    let length = std::fs::metadata(&a[2]).unwrap().len() as usize - 8;
    assert_eq!(length % size, 0, "truncated record");
    let mut record = vec![0; size];
    for _ in 0..length / size {
        input.read_exact(&mut record).unwrap();
        let mut x = [0; FEATURES];
        for (v, b) in x.iter_mut().zip(record[18..].chunks_exact(2)) {
            *v = i16::from_le_bytes(b.try_into().unwrap());
        }
        output
            .write_all(&model.evaluate_features(&x).to_le_bytes())
            .unwrap();
    }
    output.flush().unwrap();
}
