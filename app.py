
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import responder_pergunta

app = FastAPI()

class QuestionRequest(BaseModel):
    pergunta: str

@app.post("/responder")
def responder(request: QuestionRequest):
    resposta = responder_pergunta(request.pergunta)
    return {"resposta": resposta}
