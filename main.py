from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
# Zero-load startup: Imports moved inside functions
import uvicorn

app = FastAPI(title="Baby Blooms API")

# Enable CORS so your other projects can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio

_engine = None
_engine_initialized = False
_engine_error = None

async def initialize_engine_background():
    global _engine, _engine_initialized, _engine_error
    def _init():
        from rag_engine import BabyBloomEngine
        return BabyBloomEngine()
        
    try:
        print("Starting background initialization of BabyBloomEngine...")
        _engine = await asyncio.to_thread(_init)
        _engine_initialized = True
        print("BabyBloomEngine initialized successfully in background.")
    except Exception as e:
        print(f"Error initializing engine in background: {e}")
        _engine_error = str(e)

@app.on_event("startup")
async def startup_event():
    print("Scheduling BabyBloomEngine initialization...")
    asyncio.create_task(initialize_engine_background())

def get_engine():
    global _engine, _engine_initialized, _engine_error
    if _engine_error:
        raise HTTPException(status_code=500, detail=f"RAG Engine initialization failed: {_engine_error}")
    if not _engine_initialized:
        raise HTTPException(status_code=503, detail="RAG Engine is still initializing. Please try again in a few seconds.")
    return _engine

@app.get("/")
@app.head("/")
async def root():
    return {
        "message": "Baby-bloom Chatbot API is LIVE!",
        "integration": "Use POST /ask to chat, or visit /docs for API documentation.",
        "status": "Ready"
    }

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
    return {"status": "healthy", "engine_loaded": _engine is not None}

import os

if __name__ == "__main__":
    # Use the dynamic PORT assigned by Render, defaulting to 8000 for local use
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
