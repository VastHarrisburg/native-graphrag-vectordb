use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Chunk {
    pub id: String,
    pub content: String,
    pub location: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VectorEntry {
    pub chunk_id: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VectorQuery {
    pub vector: Vec<f32>,
}

pub fn convert_chunk(file: &str) -> Vec<Chunk>{
    let text = std::fs::read_to_string(file).unwrap();
    let serialized: Vec<Chunk> = serde_json::from_str(&text).unwrap();
    return serialized;
}

pub fn convert_to_json(c: &Chunk) -> String {
    let e = serde_json::to_string(&c).unwrap();
    return e;
}

pub fn convert_vector(file: &str) -> Vec<VectorEntry>{
    let text = std::fs::read_to_string(file).unwrap();
    let serialized: Vec<VectorEntry> = serde_json::from_str(&text).unwrap();
    return serialized;
}

pub fn read_query(file: &str) -> Vec<f32> {
    let text = std::fs::read_to_string(file).unwrap();
    let e: VectorQuery = serde_json::from_str(&text).unwrap();
    return e.vector;
}
