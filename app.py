
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import responder_pergunta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique ["http://127.0.0.1:5500"] se usar Live Server
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Permite todos os headers
)


class QuestionRequest(BaseModel):
    pergunta: str

@app.post("/responder")
def responder(request: QuestionRequest):
    resposta = responder_pergunta(request.pergunta)
    return {"resposta": resposta}
