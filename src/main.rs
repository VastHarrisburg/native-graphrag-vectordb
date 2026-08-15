mod vector;
mod storage;
mod search;
use serde::{Deserialize, Serialize};
use std::env;
use clap::Parser;
use std::fs;
use std::io;
use inline_python::python;
use std::path::Path;


#[derive(Parser, Debug)]
struct Cli {
    command: String,
    #[arg(default_value_t = String::from("Hello, Rust!"))]
    query: String,
    #[arg(default_value_t = 2)]
    chunk_size: i32,
    #[arg(default_value_t = 1)]
    context: i32,
    #[arg(default_value = "data/read.txt")]
    path: std::path::PathBuf
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Stats{
    documents: Vec<String>,
    chunks: u32,
    vectors: u32
}

fn main() {
    let interface = Cli::parse();


    let command = interface.command.as_str();
    let query = interface.query.as_str();
    let chunk_size = interface.chunk_size;
    let context = interface.context;
    let path = interface.path;

    match command {
        "stats" => {
            println!("Running stats");
            let c = find_documents();
            match c{
                Ok(v) => {
                    println!("The documents in data are {:?}", v);
                },
                Error => {println!("Err");}
            }
            let chunks = storage::convert_chunk("data/chunks.json");
            let chunk_length = chunks.len();
            println!("Total amount of chunks are {}", chunk_length);
            let vector = storage::convert_vector("data/vectors.json");
            let vector_length = vector.len();
            println!("Total amount of vectors are {}", vector_length);
            if(chunk_length != vector_length){
                println!("ERROR, Chunks and Vectors don't match");
            }
            
        }

        "search-vector" => {
            println!("Running vector search");
            processing(query, chunk_size, context);
            let result = search::search("data/vectors.json", "data/query_vector.json");
            println!("{:?}", result);
        }
        
        "remove-doc" => {
            match remove_file(&path) {
                Ok(()) => println!("File deleted successfully"),
                Err(error) => eprintln!("Failed to delete file: {error}"),
            }
        }

        _ => {
            eprintln!("Unknown command: {command}");
        }
    }
}

fn find_documents() -> Result<Vec<String>, io::Error> {
    let files = fs::read_dir("data")?;
    let mut v: Vec<String> = Vec::new();
    for file in files{
        let entry = file?;
        let name = match entry.file_name().into_string() {
            Ok(name) => name,
            Err(_) => {
                eprintln!("Filename is not valid UTF-8");
                continue;
            }
        };
        v.push(name);
    }
    return Ok(v);
}
fn processing(query: &str, chunk_size: i32, overlap: i32) {
    python! {
        import sys;
        query_string = 'query;
        chunk = 'chunk_size;
        over = 'overlap;
        sys.path.insert(0, "scripts");
        from processing import Processing;
        query_vector = Processing.convert_query(query_string); #put convert_query into write_query once everything works
        Processing.write_query(query_vector); #put convert_query into write_query once everything works
        Processing.write_chunks("data/read.txt", chunk, over);
        Processing.create_embeddings("data/vectors.json");
    }
}

pub fn remove_file(path: &Path) -> io::Result<()> {
    println!("Attempting to remove: {}", path.display());
    println!("Exists before deletion: {}", path.exists());

    fs::remove_file(path)?;

    println!("Exists after deletion: {}", path.exists());
    Ok(())
}




