import json
from pathlib import Path;
from sentence_transformers import SentenceTransformer
class Processing:    
    
    @classmethod
    def write_chunks(cls, filePath, chunk_size, overlap):
        chunks = cls.read_chunks(filePath, chunk_size, overlap);
        file_path = Path("data/chunks.json");
        try:
            with file_path.open("w", encoding="UTF-8") as target: 
                json.dump(chunks, target, indent=4)
        except FileNotFoundError as error:
            print("File not found")
        except json.JSONDecodeError as error:
            print("Invalid JSON:", error)
        except PermissionError as error:
            print("Permission denied:", error)
    
    @classmethod
    def create_embeddings(cls, filePath):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        filepath_chunks = Path("data/chunks.json");
        filepath_vectors = Path(filePath);
        try:
            with open(filepath_chunks, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
                print("File not found")
        except json.JSONDecodeError as error:
                print("Invalid JSON:", error)
        except PermissionError as error:
            print("Permission denied:", error)
        sentences = []
        length = len(data);
        print(length);
        for i in range(0,length):
            sentences.append(data[i]["content"])
        embeddings = model.encode(sentences)
        vectors = []
        for i in range(0, length):
            vector = {
                "chunk_id": data[i]["id"],
                "vector": embeddings[i].tolist()
            };
            vectors.append(vector);
        try:
            with filepath_vectors.open("w", encoding="UTF-8") as target: 
                json.dump(vectors, target, indent=4)
        except FileNotFoundError:
            print("File not found")
        except json.JSONDecodeError as error:
            print("Invalid JSON:", error)
        except PermissionError as error:
            print("Permission denied:", error)

    
    @classmethod
    def read_chunks(cls, filePath, chunk_size, overlap):
        file_path = Path(filePath);
        if not file_path.exists():
            raise ValueError("The file DNE");
        if file_path.is_file() and file_path.stat().st_size == 0:
            raise ValueError("The file is empty!");
        if chunk_size <= overlap:
            raise ValueError("You cannot chunk like this!");
        content = file_path.read_text(encoding="utf-8")
        sentences = [
            " ".join(s.split())
            for s in content.split(".")
            if s.strip()
        ];
        chunks = [];
        size = len(sentences);
        sentence = "";
        step = chunk_size - overlap;
        mod = size % chunk_size;
        final = 0;
        for i in range(0, size, step):
            if(i < (size-chunk_size)):
                for j in range(chunk_size):
                    sen1 = " ".join(sentences[i+j].split());
                    sentence = sentence + ". " + sen1;
                sentence = sentence + ".";
                sentence = sentence[2:];
                chunk = {
                    "id": filePath + "_chunk_" + str(i/step),
                    "content": sentence,
                    "location": filePath
                };
                final = i/step;
                sentence = "";
                chunks.append(chunk);
        for i in range((size-step), size):
            sen1 = " ".join(sentences[i].split());
            sentence = sentence + ". " + sen1;
        sentence = sentence + ".";
        sentence = sentence[2:];
        chunk = {
            "id": filePath + "_chunk_" + str(final+1),
            "content": sentence,
            "location": filePath
        };
        sentence = "";
        chunks.append(chunk);
        return chunks;
    
    @classmethod
    def convert_query(cls, query):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(query)
        query_vector = {
            "vector": embeddings.tolist()
        };
        return query_vector
    
    @classmethod
    def write_query(cls, query):
        file_path = Path("data/query_vector.json");
        try:
            with file_path.open("w", encoding="UTF-8") as target: 
                json.dump(query, target, indent=4)
        except FileNotFoundError as error:
            print("File not found")
        except json.JSONDecodeError as error:
            print("Invalid JSON:", error)
        except PermissionError as error:
            print("Permission denied:", error)
 #Reminder to create a system that manages reading and storing json for files in the future (2-3 weeks from now)
    
    
