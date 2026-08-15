from sentence_transformers import SentenceTransformer
from pathlib import Path;
from processing import Processing
import json

#query = "Why do cache misses slow down CPU performance?"
#query_vector = Processing.convert_query(query); #put convert_query into write_query once everything works
#Processing.write_query(query_vector); #put convert_query into write_query once everything works
Processing.write_chunks("data/backup.txt", 2, 1);
#Processing.create_embeddings("data/vectors.json");






