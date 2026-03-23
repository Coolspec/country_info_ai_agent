import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.config_manager import load_env
from utils.logger import get_logger
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import process_chat

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment variables on startup
    logger.info("Starting up FastAPI server. Loading environment variables...")
    load_env()
    yield
    logger.info("Shutting down FastAPI server...")

app = FastAPI(
    title="Country Info AI Agent API",
    description="An API exposing the LangGraph agent for querying country information.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:8501,http://127.0.0.1:8501"
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Submit a plain text query to the Country Info AI Agent.
    """
    logger.info("Received request to /api/chat.")
    response = await process_chat(request)
    return response

@app.get("/health")
def health_check():
    """
    Basic health check endpoint.
    """
    return {"status": "healthy"}
