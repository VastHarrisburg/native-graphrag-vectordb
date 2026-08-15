use crate::vector;
use crate::storage;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f32,
}

pub fn search(fileVectors: &str, fileQuery: &str) -> Vec<SearchResult>{
    let e = storage::convert_vector(fileVectors);
    let a = storage::read_query(fileQuery);
    let z = e.len();
    let mut top = Vec::new();
    if z < 3 {
        for i in 0..z {
            let t = vector::cosine_similarity(&a,&e[i].vector);
            match t{
                None => println!("Invalid"),
                Some(similarity) => {
                    let s = SearchResult{
                        chunk_id: e[i].chunk_id.clone(),
                        score: similarity,
                    };
                    top.push(s);
                },
            };
        }
    }
    for i in 0..z {
        let t = vector::cosine_similarity(&a,&e[i].vector);
        match t{
            None => println!("Invalid"),
            Some(similarity) => {
                let s = SearchResult{
                    chunk_id: e[i].chunk_id.clone(),
                    score: similarity,
                };
                top.push(s);
            },
        };
    }
    top.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let truncated_list: Vec<SearchResult> = top[..3].to_vec();
    return truncated_list;
}