
import os
import re
import uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# =========================================================
# CONFIGURAÇÕES
# =========================================================
load_dotenv()  # Carrega variáveis do .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "treinos"
PASTA_DADOS = "dados_rag"

# =========================================================
# Funções auxiliares (mantidas do seu script)
# =========================================================
def ler_arquivo_txt(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler arquivo {caminho}: {e}")
        return ""

def detectar_secao(texto: str) -> str:
    SECOES_REGEX = [
        ("Peitoral", r"\bPeitoral\b|\bPeito\b"),
        ("Dorsal", r"\bDorsal\b|\bCostas\b"),
        ("Deltoides", r"\bDeltoides\b|\bOmbros\b"),
        ("Quadríceps", r"\bQuadríceps\b"),
        ("Posterior da coxa", r"\bPosterior\b|\bIsquiotibiais\b"),
        ("Panturrilha", r"\bPanturrilha\b|\bGastrocnêmio\b|\bSóleo\b"),
        ("Glúteos", r"\bGlúteos\b|\bGlúteo\b"),
        ("Bíceps", r"\bBíceps\b"),
        ("Tríceps", r"\bTríceps\b"),
        ("Abdômen", r"\bAbdômen\b|\bAbdominais\b"),
    ]
    for nome, padrao in SECOES_REGEX:
        if re.search(padrao, texto, flags=re.IGNORECASE):
            return nome
    return "Geral"

def chunk_text(texto, tamanho=500, overlap=100):
    palavras = texto.split()
    chunks = []
    stride = max(1, tamanho - overlap)
    for i in range(0, len(palavras), stride):
        chunk = " ".join(palavras[i:i + tamanho])
        if chunk:
            chunks.append(chunk)
        if i + tamanho >= len(palavras):
            break
    return chunks

def processar_arquivo_otimizado(caminho):
    texto = ler_arquivo_txt(caminho)
    if not texto.strip():
        return []
    categoria = re.search(r"\[Categoria:\s*(.*?)\]", texto)
    fonte = re.search(r"\[Fonte:\s*(.*?)\]", texto)
    tipo = re.search(r"\[Tipo:\s*(.*?)\]", texto)
    categoria = categoria.group(1) if categoria else "Desconhecida"
    fonte = fonte.group(1) if fonte else "Não informado"
    tipo = tipo.group(1) if tipo else "Texto"
    chunks = re.split(r"###\s*Chunk\s*\d+:", texto)
    if len(chunks) == 1:
        chunks = chunk_text(texto)
    dataset = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50 and not chunk.startswith("[Categoria"):
            dataset.append({
                "id": str(uuid.uuid4()),
                "categoria": categoria,
                "secao": detectar_secao(chunk),
                "conteudo": chunk,
                "fonte": fonte,
                "tipo": tipo
            })
    return dataset

def criar_dataset_otimizado(pasta=PASTA_DADOS):
    dataset = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".txt"):
            caminho = os.path.join(pasta, arquivo)
            dataset.extend(processar_arquivo_otimizado(caminho))
    return dataset

def salvar_no_qdrant(dataset):
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    qdrant = QdrantClient(path="./db")
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
    pontos = []
    for item in dataset:
        emb = encoder.encode(item["conteudo"]).tolist()
        pontos.append(models.PointStruct(id=item["id"], vector=emb, payload=item))
    qdrant.upload_points(collection_name=COLLECTION_NAME, points=pontos)
    return encoder, qdrant

def rerank_hibrido(pergunta, hits_points, top_k=20):
    try:
        docs = [Document(page_content=p.payload.get("conteudo", ""), metadata=p.payload) for p in hits_points]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = top_k
        bm25_results = bm25.retrieve(pergunta)
        return bm25_results[:top_k]
    except:
        return [Document(page_content=p.payload.get("conteudo", ""), metadata=p.payload) for p in hits_points[:top_k]]

# =========================================================
# Função principal para responder pergunta
# =========================================================
def responder_pergunta(pergunta: str) -> str:
    dataset = criar_dataset_otimizado(PASTA_DADOS)
    if not dataset:
        return "Nenhum conteúdo disponível para consulta."
    encoder, qdrant = salvar_no_qdrant(dataset)
    query_vector = encoder.encode(pergunta).tolist()
    hits = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=50)
    reranked_docs = rerank_hibrido(pergunta, hits.points, top_k=20)
    contexto = "\n\n".join([d.page_content for d in reranked_docs])
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
    prompt = (
        "Você é um especialista em musculação. "
        "Com base EXCLUSIVAMENTE no contexto fornecido, responda em UM ÚNICO PARÁGRAFO, claro e direto. "
        "Se não houver dados suficientes, diga isso explicitamente.\n\n"
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA:\n{pergunta}"
    )
    resposta = llm.invoke([HumanMessage(content=prompt)])
    return resposta.content.strip()
