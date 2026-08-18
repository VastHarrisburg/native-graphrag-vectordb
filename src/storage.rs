use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Chunk {
    #[serde(alias = "id")]
    pub chunk_id: String,
    #[serde(default)]
    pub document_id: String,
    #[serde(alias = "content")]
    pub text: String,
    #[serde(default)]
    pub chunk_index: usize,
    #[serde(alias = "location")]
    pub source: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VectorEntry {
    pub chunk_id: String,
    #[serde(default)]
    pub document_id: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VectorQuery {
    pub vector: Vec<f32>,
}

fn read_json<T: serde::de::DeserializeOwned>(file: &str) -> Result<T, String> {
    let text =
        fs::read_to_string(file).map_err(|error| format!("Could not read {file}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("Could not parse {file}: {error}"))
}

pub fn convert_chunk(file: &str) -> Result<Vec<Chunk>, String> {
    read_json(file)
}

pub fn convert_vector(file: &str) -> Result<Vec<VectorEntry>, String> {
    read_json(file)
}

pub fn read_query(file: &str) -> Result<Vec<f32>, String> {
    let query: VectorQuery = read_json(file)?;
    Ok(query.vector)
}
