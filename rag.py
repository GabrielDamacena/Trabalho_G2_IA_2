
import os
import re
import requests
from bs4 import BeautifulSoup
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# =========================================================
# CONFIGURAÇÕES
# =========================================================
GOOGLE_API_KEY   = ""   # Coloque sua API key
GEMINI_MODEL     = "gemini-2.5-flash"     # Modelo estável
COLLECTION_NAME  = "treinos"

# =========================================================
# LINKS E PDFs (ajuste conforme sua necessidade)
# =========================================================
categorias_links = {
    "Estudos sobre Hipertrofia": [
        "https://pubmed.ncbi.nlm.nih.gov/37414459/",
        "https://pubmed.ncbi.nlm.nih.gov/33497853/",
        "https://ferronaveia.com.br/divisao-de-treino-tipos-e-como-montar-na-pratica"
    ],
    "Divisões de treino": [
        "https://pubmed.ncbi.nlm.nih.gov/34468591/",
        "https://pubmed.ncbi.nlm.nih.gov/38874955/",
    ]
}
pdf_urls_por_categoria = {
    # "Melhores Exercícios (PDFs)": [
    #     "https://meu-servidor.com/melhores_exercicios.pdf",
    # ]
}
pdf_locais_por_categoria = {
    # Exemplo: inclua aqui seus PDFs locais
    "Melhores Exercícios (Local)": ["melhores_exercicios.txt"],
    "Glúteos (Local)": ["exercicios_gluteos.txt"],
}

# =========================================================
# DETECÇÃO DE SEÇÃO (metadados)
# =========================================================
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
def detectar_secao(texto: str) -> str:
    for nome, padrao in SECOES_REGEX:
        if re.search(padrao, texto, flags=re.IGNORECASE):
            return nome
    return "Geral"

# =========================================================
# EXTRAÇÃO DE TEXTO: WEB (HTML)
# =========================================================
def baixar_conteudo(url):
    """Baixa conteúdo HTML simples via requests + BeautifulSoup."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style"]): s.extract()
        texto = " ".join(soup.stripped_strings)
        return texto.strip()
    except Exception as e:
        print(f"Erro ao baixar {url}: {e}")
        return ""

# =========================================================
# EXTRAÇÃO DE TEXTO: PDF (local e URL)
# =========================================================
def extrair_texto_pdf_local(caminho_pdf: str) -> str:
    """
    Extrai texto de um PDF local.
    Tenta PyPDFLoader (LangChain); se indisponível, usa pdfplumber.
    Faz uma verificação de extração (opcional) para garantir que a seção 'Peitoral' está presente.
    """
    texto = ""
    docs_text = ""

    # 1) Tenta PyPDFLoader
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(caminho_pdf)
        docs = loader.load()
        if docs:
            texto = "\n".join([d.page_content or "" for d in docs])
            docs_text = texto  # texto completo para verificação
    except Exception as e:
        print(f"[PyPDFLoader indisponível ou falhou] {e} → usando pdfplumber.")

    # 2) Fallback: pdfplumber (caso texto vazio ou falha)
    if not texto.strip():
        try:
            with pdfplumber.open(caminho_pdf) as pdf:
                for p in pdf.pages:
                    t = p.extract_text() or ""
                    texto += t + "\n"
                    docs_text += t
        except Exception as e:
            print(f"[pdfplumber falhou] {caminho_pdf}: {e}")
            return ""

    # ✅ Confirmação de extração (opcional; ative quando o PDF *dever* conter a seção Peitoral)
    try:
        assert "Peitoral" in docs_text, "Seção 'Peitoral' não encontrada na extração do PDF local."
    except AssertionError as ae:
        print(f"[EXTRAÇÃO INCOMPLETA] {ae}")
        # Se quiser interromper o pipeline ao faltar 'Peitoral', descomente:
        # return ""

    return texto.strip()

def extrair_texto_pdf_url(url_pdf: str) -> str:
    """
    Faz download do PDF e extrai seu texto (usando a mesma estratégia de local).
    """
    try:
        r = requests.get(url_pdf, timeout=60)
        r.raise_for_status()
        tmp = "tmp_ingest.pdf"
        with open(tmp, "wb") as f:
            f.write(r.content)
        texto = extrair_texto_pdf_local(tmp)
        os.remove(tmp)
        return texto
    except Exception as e:
        print(f"Erro ao baixar PDF {url_pdf}: {e}")
        return ""

# =========================================================
# CHUNKING MAIOR COM OVERLAP (1200 / 150)
# =========================================================
def chunk_text(texto, tamanho=1200, overlap=150):
    """
    Divide o texto em chunks com overlap para preservar cabeçalho + corpo.
    tamanho: número aproximado de palavras por chunk
    overlap: número de palavras de sobreposição entre chunks consecutivos
    """
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

# =========================================================
# LEITURA DE ARQUIVOS TXT (opcional)
# =========================================================
def ler_arquivo_txt(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler arquivo {caminho}: {e}")
        return ""

# =========================================================
# DATASET (web + PDFs URL + PDFs locais + TXT)
# =========================================================
def criar_dataset(categorias_links,
                  pdf_urls_por_categoria=None,
                  pdf_locais_por_categoria=None,
                  arquivos_locais=None):
    dataset = []
    id_counter = 1

    # 1) WEB (HTML)
    for categoria, links in categorias_links.items():
        for link in links:
            conteudo = baixar_conteudo(link)
            if conteudo:
                for chunk in chunk_text(conteudo):
                    dataset.append({
                        "id": id_counter,
                        "categoria": categoria,
                        "secao": detectar_secao(chunk),
                        "conteudo": chunk,
                        "fonte": link,
                        "tipo": "html"
                    })
                    id_counter += 1

    # 2) PDFs por URL
    if pdf_urls_por_categoria:
        for categoria, urls in pdf_urls_por_categoria.items():
            for url_pdf in urls:
                texto_pdf = extrair_texto_pdf_url(url_pdf)
                if texto_pdf:
                    for chunk in chunk_text(texto_pdf):
                        dataset.append({
                            "id": id_counter,
                            "categoria": categoria,
                            "secao": detectar_secao(chunk),
                            "conteudo": chunk,
                            "fonte": url_pdf,
                            "tipo": "pdf_url"
                        })
                        id_counter += 1

    # 3) PDFs locais
    if pdf_locais_por_categoria:
        for categoria, caminhos in pdf_locais_por_categoria.items():
            for caminho_pdf in caminhos:
                texto_pdf = extrair_texto_pdf_local(caminho_pdf)
                if texto_pdf:
                    for chunk in chunk_text(texto_pdf):
                        dataset.append({
                            "id": id_counter,
                            "categoria": categoria,
                            "secao": detectar_secao(chunk),
                            "conteudo": chunk,
                            "fonte": caminho_pdf,
                            "tipo": "pdf_local"
                        })
                        id_counter += 1

    # 4) TXT locais (se necessário)
    if arquivos_locais:
        for categoria, caminho in arquivos_locais.items():
            conteudo = ler_arquivo_txt(caminho)
            if conteudo:
                for chunk in chunk_text(conteudo):
                    dataset.append({
                        "id": id_counter,
                        "categoria": categoria,
                        "secao": detectar_secao(chunk),
                        "conteudo": chunk,
                        "fonte": caminho,
                        "tipo": "txt_local"
                    })
                    id_counter += 1

    return dataset

# =========================================================
# INDEXAÇÃO EM QDRANT
# =========================================================
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
    print(f"{len(dataset)} chunks salvos no Qdrant.")
    return encoder, qdrant

# =========================================================
# RERANKING HÍBRIDO: BM25 + Vetorial (fallback automático)
# =========================================================
def rerank_hibrido_bm25(pergunta, hits_points, top_k=3):
    """
    Reranking simples:
    - Primeiro usa BM25 para ordenar por relevância lexical.
    - Em seguida devolve os top_k da BM25.
    - Fallback: se BM25 não disponível, devolve os primeiros pontos originais.
    """
    try:

        docs = [Document(page_content=p.payload.get("conteudo", ""), metadata=p.payload) for p in hits_points]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = max(top_k, 5)

        bm25_results = bm25.get_relevant_documents(pergunta)
        reranked = bm25_results[:top_k]

        out_points_like = []
        for d in reranked:
            out_points_like.append({"payload": d.metadata | {"conteudo": d.page_content}})
        return out_points_like

    except Exception as e:
        print(f"[BM25 indisponível] usando vetorial puro. Motivo: {e}")
        return [{"payload": p.payload} for p in hits_points[:top_k]]

# =========================================================
# CONSULTA (SEM FILTRO DE SEÇÃO) + RERANKING
# =========================================================
def consulta_rag(pergunta, encoder, qdrant, limit_busca=10, top_k_contexto=3):
    # Vetor da query
    query_vector = encoder.encode(pergunta).tolist()

    # Busca ampla (sem filtro por 'secao')
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit_busca
    )

    # Reranking híbrido para refinar contexto
    reranked_points = rerank_hibrido_bm25(pergunta, hits.points, top_k=top_k_contexto)
    contexto = "\n\n".join([p["payload"]["conteudo"] for p in reranked_points])

    # LLM
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)

    prompt = (
        "Você é um especialista em musculação. "
        "Com base EXCLUSIVAMENTE no contexto fornecido, responda em UM ÚNICO PARÁGRAFO, claro e direto. "
        "Se não houver dados suficientes no contexto, diga isso explicitamente.\n\n"
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA:\n{pergunta}"
    )

    resposta = llm.invoke([HumanMessage(content=prompt)])
    print("\n=== Resposta com RAG ===")
    print(resposta.content.strip())

# =========================================================
# MAIN
# =========================================================

def main():
    arquivos_locais = {
        "Melhores Exercícios (TXT)": "melhores_exercicios.txt",
        "Glúteos (TXT)": "exercicios_gluteos.txt",
    }

    dataset = criar_dataset(
        categorias_links,
        pdf_urls_por_categoria=None,  # Não estamos usando PDFs online
        pdf_locais_por_categoria=None,  # Não estamos usando PDFs locais
        arquivos_locais=arquivos_locais  # ✅ Agora processa TXT
    )

    if dataset:
        encoder, qdrant = salvar_no_qdrant(dataset)

        # Exemplo de consulta
        consulta_rag(
            "Quais são os melhores exercícios para glúteos?",
            encoder, qdrant,
            limit_busca=15,
            top_k_contexto=3
        )

        consulta_rag(
            "Quais são os melhores exercícios para peitoral visando hipertrofia?",
            encoder, qdrant,
            limit_busca=15,
            top_k_contexto=3
        )
    else:
        print("Nenhum conteúdo foi processado.")

if __name__ == "__main__":
    main()
