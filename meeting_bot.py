from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from bson import ObjectId
import openai
import numpy as np
import faiss
import os
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load OpenAI key once 🔧
with open("gpt-key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)  # 🔧 move to global scope


# MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["meeting_notes_db"]
collection = db["meeting_notes"]




# FAISS setup for semantic search
dimension = 1536  # embedding dimension for text-embedding-3-small
faiss_index = faiss.IndexFlatL2(dimension)
id_mapping = {}


# Load existing FAISS index if available
if os.path.exists("faiss_index.bin"):
    faiss_index = faiss.read_index("faiss_index.bin")
    print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")

# Load ID mapping
if os.path.exists("faiss_id_map.txt"):
    with open("faiss_id_map.txt", "r") as f:
        for line in f:
            idx, mongo_id = line.strip().split(",")
            id_mapping[int(idx)] = mongo_id



# Data models
class MeetingNote(BaseModel):
    title: str
    date: str
    department: str
    summary: str
    meeting_url: str

class SearchQuery(BaseModel):
    query: str
    

@app.post("/upload")
async def upload_meeting_note(note: MeetingNote):
    try:
        # Step 1: Generate embedding from summary
        embed_response = client.embeddings.create(
            input=note.summary,
            model="text-embedding-3-small"
        )
        embedding = embed_response.data[0].embedding
        vector = np.array([embedding], dtype='float32')

        # Step 2: Ask GPT-4 to extract tags from the summary
        tag_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts up to 5 relevant, short, lowercase tags for categorizing meeting notes. Return tags as a JSON array of strings."},
                {"role": "user", "content": f"Generate tags for this meeting summary:\n\n{note.summary}"}
            ],
            temperature=0.3
        )
        tags = eval(tag_response.choices[0].message.content.strip())
        print(f"Extracted tags: {tags}")

        # Step 3: Prepare MongoDB record
        mongo_record = note.dict()
        mongo_record["tags"] = tags

        # Step 4: Save to MongoDB
        inserted = collection.insert_one(mongo_record)
        doc_id = str(inserted.inserted_id)

        # Step 5: Save vector to FAISS
        faiss_index.add(vector)
        id_mapping[faiss_index.ntotal - 1] = doc_id

        # ✅ Save FAISS index to disk
        faiss.write_index(faiss_index, "faiss_index.bin")

        # ✅ Save ID map to text file
        with open("faiss_id_map.txt", "w") as f:
            for idx, mongo_id in id_mapping.items():
                f.write(f"{idx},{mongo_id}\n")


        # Optional: persist FAISS + ID map here if you're doing that

        return {"status": "success", "id": doc_id, "tags": tags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/search-and-summarize")
async def search_and_summarize(payload: SearchQuery):
    try:
        # Generate embedding using the new client
        embed_response = client.embeddings.create(
            input=payload.query,
            model="text-embedding-3-small"
        )
        query_vector = np.array([embed_response.data[0].embedding], dtype='float32')

        # Semantic search in FAISS
        top_k = 5
        distances, indices = faiss_index.search(query_vector, top_k)
        matched_ids = [id_mapping.get(i) for i in indices[0] if i in id_mapping]

        # Fetch from MongoDB
        matched_docs = []
        for doc_id in matched_ids:
            doc = collection.find_one({"_id": ObjectId(doc_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
                matched_docs.append(doc)

        if not matched_docs:
            return {"summary": "No relevant meeting notes found."}

        # Concatenate matched summaries
        combined_text = "\n\n".join([
            f"{doc['title']} ({doc['date']}): {doc['summary']}" for doc in matched_docs
        ])

        # Generate summary using GPT-4 via new client
        summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question strictly using the content from the provided meeting notes. Do not guess or make up answers."},
            {"role": "user", "content": f"Question: {payload.query}\n\nMeeting Notes:\n{combined_text}"}
        ],
        temperature=0.3
)


        summary_text = summary_response.choices[0].message.content
        return {"summary": summary_text, "matched_notes": matched_docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/debug/faiss")
def get_faiss_count():
    return {
        "total_faiss_vectors": faiss_index.ntotal,
        "mapped_ids": list(id_mapping.values())
    }

