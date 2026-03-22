from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag_engine import BabyBloomEngine
import uvicorn

app = FastAPI(title="Baby Blooms API")

# Enable CORS so your other projects can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine lazily to avoid startup timeouts on Render
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        print("Initializing BabyBloomEngine (this may take a minute)...")
        try:
            _engine = BabyBloomEngine()
        except Exception as e:
            print(f"Error initializing engine: {e}")
            _engine = None # Ensure _engine remains None if initialization fails
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG Engine: {e}")
    return _engine

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []

@app.post("/ask")
async def ask_baby_bloom(request: ChatRequest):
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized.")
    
    try:
        answer, intent = engine.ask(request.question, request.chat_history)
        return {"answer": answer, "intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
