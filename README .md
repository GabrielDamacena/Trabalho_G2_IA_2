# Projeto: Sistema RAG para Recomendação de Treinos de Hipertrofia

## 1. Problema
Pessoas que treinam para hipertrofia enfrentam dificuldade em encontrar treinos personalizados e confiáveis. A maioria dos conteúdos disponíveis é genérica, não considera equipamentos disponíveis e pode levar a resultados insatisfatórios ou até lesões.

## 2. Abordagem
Este projeto utiliza **RAG (Retrieval-Augmented Generation)** para fornecer respostas contextualizadas sobre treinos de musculação, com foco em hipertrofia.
A solução combina:
- **Banco vetorial (Qdrant)** para armazenar treinos e conteúdos relevantes.
- **SentenceTransformer** para gerar embeddings semânticos.
- **Gemini (Google Generative AI)** para gerar respostas claras e contextualizadas.
- **FastAPI** para disponibilizar uma API simples e escalável.

## 3. Arquitetura
```
Usuário → FastAPI (app.py) → Função responder_pergunta (rag_pipeline.py)
       → Busca no Qdrant (similaridade semântica)
       → Re-ranking híbrido (BM25 + embeddings)
       → Geração de resposta via LLM (Gemini)
```

**Componentes principais:**
- `app.py`: API REST com endpoint `/responder`.
- `rag_pipeline.py`: Pipeline RAG (pré-processamento, indexação, busca e geração).

## 4. Como Executar o Projeto

### Passo 1: Preparar Ambiente
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Instalar dependências
pip install -r requirements.txt
```

**Dependências principais:**
- fastapi
- uvicorn
- sentence-transformers
- qdrant-client
- langchain-google-genai
- python-dotenv

### Passo 2: Configurar Variáveis
Crie um arquivo `.env` com:
```
GOOGLE_API_KEY=YOUR_API_KEY
```

### Passo 3: Preparar Dataset
- Crie a pasta `dados_rag` e adicione arquivos `.txt` com treinos.
- Cada arquivo pode conter:
```
[Categoria: Hipertrofia]
[Tipo: Exercício]
[Fonte: Autor]
### Chunk 1:
Conteúdo do treino...
```

### Passo 4: Executar API
```bash
uvicorn app:app --reload
```
Acesse:
`POST http://127.0.0.1:8000/responder`
Body:
```json
{
  "pergunta": "Qual treino para peitoral com halteres?"
}
```

## 5. Exemplos de Resultados
**Input:**
"Sugira treino para bíceps."

**Output:**
```json
{
  "resposta": "Execute rosca direta com barra, 3 séries de 8-12 repetições, mantendo postura correta."
}
```

## 6. Front-End
Este projeto também inclui uma interface web simples para interação com a API.

**Arquivos principais:**
- `index.html`: Estrutura da página, contendo o cabeçalho, área de chat e campo de entrada.
- `style.css`: Responsável pelo design responsivo, cores, espaçamento e animações (inclui spinner para carregamento).
- `script.js`: Lógica do front-end, incluindo:
  - Captura da entrada do usuário.
  - Exibição das mensagens no chat.
  - Requisição à API FastAPI (`/responder`) usando `fetch`.
  - Indicador de carregamento enquanto a IA processa a resposta.

**Integração com a API:**
- O front-end envia a pergunta do usuário via POST para `http://127.0.0.1:8000/responder`.
- Recebe a resposta da IA e exibe no chat.

### Como executar o front-end:
1. Abra `index.html` em um navegador (pode usar Live Server no VSCode para melhor experiência).
2. Certifique-se de que a API está rodando (`uvicorn app:app --reload`).
3. Digite sua pergunta e veja a resposta aparecer no chat.

**Exemplo de fluxo:**
- Usuário: "Qual treino para tríceps com halteres?"
- Bot: "Execute tríceps francês com halteres, 3 séries de 10-12 repetições, mantendo cotovelos fixos."

## 7. Visualizações
- Exemplo de resposta no front-end: 
`img/print_1.png`
`img/print_2.png`
`img/print_3.png`

## 8. Próximos Passos
- Adicionar equivalencia de exercicíos 
- Integrar com app de treino.
- Expandir para outros objetivos (emagrecimento, resistência).