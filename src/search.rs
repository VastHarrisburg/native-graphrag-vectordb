use crate::storage;
use crate::vector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_id: String,
    pub text: String,
    pub similarity_score: f32,
    pub source: String,
}

pub fn search(
    vectors_file: &str,
    query_file: &str,
    chunks_file: &str,
    top_k: usize,
) -> Result<Vec<SearchResult>, String> {
    let vectors = storage::convert_vector(vectors_file)?;
    let query = storage::read_query(query_file)?;
    let chunks = storage::convert_chunk(chunks_file)?;
    let chunks_by_id: HashMap<_, _> = chunks
        .into_iter()
        .map(|chunk| (chunk.chunk_id.clone(), chunk))
        .collect();
    let mut results = Vec::new();

    for entry in vectors {
        let Some(chunk) = chunks_by_id.get(&entry.chunk_id) else {
            continue;
        };
        let Some(similarity_score) = vector::cosine_similarity(&query, &entry.vector) else {
            continue;
        };
        let document_id = if entry.document_id.is_empty() {
            chunk.document_id.clone()
        } else {
            entry.document_id
        };

        results.push(SearchResult {
            chunk_id: entry.chunk_id,
            document_id,
            text: chunk.text.clone(),
            similarity_score,
            source: chunk.source.clone(),
        });
    }

    results.sort_by(|left, right| {
        right
            .similarity_score
            .partial_cmp(&left.similarity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn search_returns_available_results_when_fewer_than_top_k() {
        let directory =
            std::env::temp_dir().join(format!("graphrag-search-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let chunks = directory.join("chunks.json");
        let vectors = directory.join("vectors.json");
        let query = directory.join("query.json");
        fs::write(
            &chunks,
            r#"[{"chunk_id":"doc_001_chunk_000","document_id":"doc_001","text":"cache","chunk_index":0,"source":"sample.txt"}]"#,
        )
        .unwrap();
        fs::write(
            &vectors,
            r#"[{"chunk_id":"doc_001_chunk_000","document_id":"doc_001","vector":[1.0,0.0]}]"#,
        )
        .unwrap();
        fs::write(&query, r#"{"vector":[1.0,0.0]}"#).unwrap();

        let results = search(
            vectors.to_str().unwrap(),
            query.to_str().unwrap(),
            chunks.to_str().unwrap(),
            5,
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc_001");
        assert_eq!(results[0].source, "sample.txt");
    }
}
