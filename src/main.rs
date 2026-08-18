mod search;
mod storage;
mod vector;

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Native Rust vector search with document-specific GraphRAG")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, ValueEnum)]
enum EmbeddingBackend {
    Auto,
    SentenceTransformers,
    Hash,
}

impl EmbeddingBackend {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::SentenceTransformers => "sentence-transformers",
            Self::Hash => "hash",
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum GraphBackend {
    Heuristic,
    Openai,
}

impl GraphBackend {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::Openai => "openai",
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Ingest or update one document and its graph.
    #[command(visible_alias = "upload")]
    Ingest {
        path: PathBuf,
        #[arg(long, default_value_t = 5)]
        chunk_size: usize,
        #[arg(long, default_value_t = 1)]
        overlap: usize,
        #[arg(long, value_enum, default_value = "auto")]
        embedding_backend: EmbeddingBackend,
        #[arg(long, value_enum, default_value = "heuristic")]
        graph_backend: GraphBackend,
        #[arg(long)]
        force: bool,
    },
    /// Run native semantic search, graph traversal, and grounded answering.
    Query {
        question: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 3)]
        max_documents: usize,
        #[arg(long, default_value_t = 8)]
        max_context: usize,
        #[arg(long, value_enum, default_value = "auto")]
        embedding_backend: EmbeddingBackend,
        #[arg(long)]
        semantic_only: bool,
        #[arg(long)]
        no_llm: bool,
    },
    /// Remove a document and all of its stored data.
    RemoveDoc { document: String },
    /// Print index and registry counts.
    Stats,
    /// Search already-created vectors without graph or answer generation.
    SearchVector {
        #[arg(long, default_value_t = 3)]
        top_k: usize,
    },
    /// Compare semantic-only and hybrid evidence retrieval.
    Benchmark {
        #[arg(default_value = "evaluation/questions.json")]
        path: PathBuf,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, value_enum, default_value = "auto")]
        embedding_backend: EmbeddingBackend,
    },
}

#[derive(Debug, Deserialize)]
struct EvaluationCase {
    query: String,
    #[serde(default)]
    expected_chunks: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EvaluationResult {
    query: String,
    mode: String,
    expected_chunks: Vec<String>,
    returned_chunks: Vec<String>,
    recall: f64,
    precision: f64,
    vector_search_ms: f64,
    graph_traversal_ms: f64,
    total_ms: f64,
    context_characters: usize,
}

fn python_executable() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn run_python(arguments: &[String]) -> Result<String, String> {
    let output = Command::new(python_executable())
        .arg("scripts/pipeline.py")
        .args(arguments)
        .output()
        .map_err(|error| format!("Could not start Python: {error}"))?;

    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if error.is_empty() {
            "The Python pipeline failed without an error message".to_string()
        } else {
            error
        });
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(value)
        .map_err(|error| format!("Could not serialize {}: {error}", path.display()))?;
    fs::write(path, format!("{serialized}\n"))
        .map_err(|error| format!("Could not write {}: {error}", path.display()))
}

fn run_query(
    question: &str,
    top_k: usize,
    max_documents: usize,
    max_context: usize,
    embedding_backend: &EmbeddingBackend,
    semantic_only: bool,
    no_llm: bool,
) -> Result<Value, String> {
    if question.trim().is_empty() {
        return Err("The question cannot be empty".to_string());
    }

    let total_started = Instant::now();
    run_python(&[
        "prepare-query".to_string(),
        question.to_string(),
        "--embedding-backend".to_string(),
        embedding_backend.as_str().to_string(),
    ])?;

    let vector_started = Instant::now();
    let semantic_results = search::search(
        "data/vectors.json",
        "data/query_vector.json",
        "data/chunks.json",
        top_k,
    )?;
    let vector_ms = vector_started.elapsed().as_secs_f64() * 1000.0;
    let semantic_path = Path::new("data/last_semantic_results.json");
    write_json(semantic_path, &semantic_results)?;

    let mut arguments = vec![
        "complete-query".to_string(),
        question.to_string(),
        "--semantic-results".to_string(),
        semantic_path.to_string_lossy().to_string(),
        "--max-documents".to_string(),
        max_documents.to_string(),
        "--max-context".to_string(),
        max_context.to_string(),
    ];

    if semantic_only {
        arguments.push("--semantic-only".to_string());
    }
    if no_llm {
        arguments.push("--no-llm".to_string());
    }

    let output = run_python(&arguments)?;
    let mut result: Value = serde_json::from_str(&output)
        .map_err(|error| format!("Could not parse query result: {error}"))?;
    result["timing_ms"]["vector_search"] = Value::from(vector_ms);
    result["timing_ms"]["total"] = Value::from(total_started.elapsed().as_secs_f64() * 1000.0);
    Ok(result)
}

fn evaluate_result(case: &EvaluationCase, mode: &str, result: &Value) -> EvaluationResult {
    let returned_chunks: Vec<String> = result["combined_context"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["chunk_id"].as_str().map(str::to_string))
        .collect();
    let expected: HashSet<_> = case.expected_chunks.iter().collect();
    let returned: HashSet<_> = returned_chunks.iter().collect();
    let matches = expected.intersection(&returned).count();
    let recall = if expected.is_empty() {
        1.0
    } else {
        matches as f64 / expected.len() as f64
    };
    let precision = if returned.is_empty() {
        if expected.is_empty() { 1.0 } else { 0.0 }
    } else {
        matches as f64 / returned.len() as f64
    };
    let context_characters = result["combined_context"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["text"].as_str())
        .map(str::len)
        .sum();

    EvaluationResult {
        query: case.query.clone(),
        mode: mode.to_string(),
        expected_chunks: case.expected_chunks.clone(),
        returned_chunks,
        recall,
        precision,
        vector_search_ms: result["timing_ms"]["vector_search"].as_f64().unwrap_or(0.0),
        graph_traversal_ms: result["timing_ms"]["graph_traversal"]
            .as_f64()
            .unwrap_or(0.0),
        total_ms: result["timing_ms"]["total"].as_f64().unwrap_or(0.0),
        context_characters,
    }
}

fn run_benchmark(
    path: &Path,
    top_k: usize,
    embedding_backend: &EmbeddingBackend,
) -> Result<Value, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    let cases: Vec<EvaluationCase> = serde_json::from_str(&contents)
        .map_err(|error| format!("Could not parse {}: {error}", path.display()))?;
    let mut results = Vec::new();

    for case in &cases {
        for (mode, semantic_only) in [("semantic", true), ("hybrid", false)] {
            let result = run_query(
                &case.query,
                top_k,
                3,
                8,
                embedding_backend,
                semantic_only,
                true,
            )?;
            results.push(evaluate_result(case, mode, &result));
        }
    }

    let summaries = ["semantic", "hybrid"].into_iter().map(|mode| {
        let selected: Vec<_> = results.iter().filter(|item| item.mode == mode).collect();
        let count = selected.len().max(1) as f64;
        serde_json::json!({
            "mode": mode,
            "queries": selected.len(),
            "mean_recall": selected.iter().map(|item| item.recall).sum::<f64>() / count,
            "mean_precision": selected.iter().map(|item| item.precision).sum::<f64>() / count,
            "mean_vector_search_ms": selected.iter().map(|item| item.vector_search_ms).sum::<f64>() / count,
            "mean_graph_traversal_ms": selected.iter().map(|item| item.graph_traversal_ms).sum::<f64>() / count,
            "mean_total_ms": selected.iter().map(|item| item.total_ms).sum::<f64>() / count,
            "mean_context_characters": selected.iter().map(|item| item.context_characters as f64).sum::<f64>() / count,
        })
    }).collect::<Vec<_>>();

    Ok(serde_json::json!({"summary": summaries, "results": results}))
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Ingest {
            path,
            chunk_size,
            overlap,
            embedding_backend,
            graph_backend,
            force,
        } => {
            let mut arguments = vec![
                "ingest".to_string(),
                path.to_string_lossy().to_string(),
                "--chunk-size".to_string(),
                chunk_size.to_string(),
                "--overlap".to_string(),
                overlap.to_string(),
                "--embedding-backend".to_string(),
                embedding_backend.as_str().to_string(),
                "--graph-backend".to_string(),
                graph_backend.as_str().to_string(),
            ];

            if force {
                arguments.push("--force".to_string());
            }

            run_python(&arguments)
        }
        Commands::Query {
            question,
            top_k,
            max_documents,
            max_context,
            embedding_backend,
            semantic_only,
            no_llm,
        } => run_query(
            &question,
            top_k,
            max_documents,
            max_context,
            &embedding_backend,
            semantic_only,
            no_llm,
        )
        .and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|error| format!("Could not print query result: {error}"))
        }),
        Commands::RemoveDoc { document } => run_python(&["remove".to_string(), document]),
        Commands::Stats => run_python(&["stats".to_string()]),
        Commands::SearchVector { top_k } => search::search(
            "data/vectors.json",
            "data/query_vector.json",
            "data/chunks.json",
            top_k,
        )
        .and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|error| format!("Could not print search results: {error}"))
        }),
        Commands::Benchmark {
            path,
            top_k,
            embedding_backend,
        } => run_benchmark(&path, top_k, &embedding_backend).and_then(|value| {
            serde_json::to_string_pretty(&value)
                .map_err(|error| format!("Could not print benchmark results: {error}"))
        }),
    };

    match result {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn upload_alias_uses_the_ingestion_pipeline() {
        let cli = Cli::try_parse_from([
            "native-graphrag-vectordb",
            "upload",
            "documents/example.txt",
            "--embedding-backend",
            "hash",
        ])
        .expect("upload should be accepted as an ingest alias");

        match cli.command {
            Commands::Ingest {
                path,
                embedding_backend: EmbeddingBackend::Hash,
                ..
            } => assert_eq!(path, PathBuf::from("documents/example.txt")),
            _ => panic!("upload should resolve to the ingest command"),
        }
    }
}
