import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot import ChatbotResponse, consultar_chatbot_veterinario


load_dotenv()

app = FastAPI(
    title="4UPet Chatbot API",
    version="1.0.0",
    description="API do assistente veterinário RAG do 4UPet.",
)


# ============================================================================
# CORS
# ============================================================================

cors_config = os.getenv("CORS_ORIGINS", "*").strip()

if cors_config == "*":
    origins = ["*"]
    allow_credentials = False
else:
    origins = [
        origem.strip()
        for origem in cors_config.split(",")
        if origem.strip()
    ]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# MODELOS DA API
# ============================================================================

class ChatRequest(BaseModel):
    mensagem: str = Field(
        min_length=1,
        max_length=2000,
    )


# ============================================================================
# ROTAS
# ============================================================================

@app.get("/")
def home():
    return {
        "mensagem": "API do chatbot 4UPet funcionando."
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/chat",
    response_model=ChatbotResponse,
)
def chat(request: ChatRequest):
    return consultar_chatbot_veterinario(
        request.mensagem
    )
