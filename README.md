# Career Intelligence Assistant

A full-stack RAG application that lets candidates upload their CV and job descriptions, then ask natural language questions about fit, skill gaps, and interview preparation.

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <repo-url> career-intel && cd career-intel

# 2. Add your API keys
cp .env.example .env
# Open .env and fill in:
#   OPENAI_API_KEY   — required for embeddings and the LLM
#   LANGCHAIN_API_KEY — optional, only needed if you want LangSmith tracing

# 3. Install dependencies
make install
# or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Run the app
make run
# or:
streamlit run app.py
```

The app opens at http://localhost:8501.

**To run with Docker:**
```bash
make docker-build
make docker-run
```

**To run tests:**
```bash
make test
```

**To run evals (requires OpenAI key):**
```bash
make evals
```

**To run the MCP server (Claude Desktop / Cursor):**
```bash
make mcp
```

Or add directly to Claude Desktop's `mcp_servers` config:
```json
{
  "career-intel": {
    "command": "python",
    "args": ["-m", "src.mcp_server"],
    "cwd": "/path/to/career-intel"
  }
}
```

The MCP server exposes five tools: `upload_cv`, `upload_job_description`, `ask`, `list_loaded_documents`, and `reset_documents`. Any MCP-compatible client can use these to query career fit and skill gaps without going through the Streamlit UI.

---

## Architecture

```mermaid
flowchart TD
    User([User / Browser]) -->|Upload CV PDF| Streamlit
    User -->|Upload JD PDF/TXT| Streamlit
    User -->|Ask question| Streamlit

    subgraph app ["app.py (Streamlit UI)"]
        Streamlit[Session State + Chat UI]
    end

    subgraph ingestion ["src/ingestion/"]
        Parser[parser.py\npypdf → LangChain Documents]
        Chunker[chunker.py\nRecursiveCharacterTextSplitter\n500 chars / 50 overlap]
        Parser --> Chunker
    end

    subgraph rag ["src/rag/"]
        VectorStore[vectorstore.py\nChromaDB\n2 collections: cv + jd\nPersisted to ./chroma_db]
        Retriever[retriever.py\nSimilarity search\nwith rank-based scoring]
        VectorStore --> Retriever
    end

    subgraph qa ["src/qa/"]
        Schemas[schemas.py\nPydantic: CareerAnswer\nSkillGap]
        Chain[chain.py\nChatOpenAI gpt-4o-mini\n.with_structured_output\nCareerAnswer]
        Schemas --> Chain
    end

    Streamlit --> Parser
    Chunker --> VectorStore
    Streamlit --> Retriever
    Retriever --> Chain
    Chain --> Streamlit

    OpenAI([OpenAI API\ngpt-4o-mini\ntext-embedding-3-small]) <--> VectorStore
    OpenAI <--> Chain
    LangSmith([LangSmith\nTracing]) -.->|traces| Chain
```

---

## RAG and LLM Approach

### Document Ingestion
- **PDF parsing**: `pypdf` extracts text page-by-page. Raises `ValueError` on image-only PDFs rather than silently returning empty context.
- **Chunking**: `RecursiveCharacterTextSplitter` at 500 characters with 50-char overlap. Chosen over token-based splitting because it preserves sentence and paragraph boundaries without requiring a tokeniser at ingest time.

### Embedding and Retrieval
- **Model**: `text-embedding-3-small` — OpenAI's best price/quality tradeoff for semantic search. 1536 dimensions.
- **Vector DB**: ChromaDB with disk persistence at `./chroma_db`. Two separate collections (`cv_documents`, `jd_documents`) so retrieval can be scoped or merged depending on the question.
- **Retrieval strategy**: `similarity_search` returns the top-k closest chunks by ChromaDB's internal distance metric. Results are assigned rank-based scores (1.0 for rank 1, decaying by position) rather than raw cosine similarity, because ChromaDB with L2 distance returns scores outside [0,1] — including negative values — which makes threshold filtering unreliable.
- **Top-K**: 6 chunks across both collections, merged and sorted by score.

### Prompt Engineering

The system prompt went through a few iterations. The first version just said "answer questions about the CV and JD" — too vague. The LLM kept returning freeform text even when structured output was configured, because the prompt didn't specify what each field was for.

The final prompt does three things explicitly:
1. Defines what each output field means (answer, confidence, skill_gaps, sources, follow_up_questions) so the LLM populates them intentionally, not by guessing
2. Instructs the LLM to tie confidence to how well the *context* supports the answer, not general certainty — this prevents confident-but-hallucinated responses
3. Includes a scope guardrail that catches off-topic questions before they reach the retriever

### Context Management

Top-k is set to 6 chunks across both collections. At 500 characters per chunk, that's roughly 3,000 characters of context — well within gpt-4o-mini's 128k context window, leaving plenty of room for the system prompt and structured output schema. The two-collection design means context is always a mix of CV and JD chunks rather than dominated by whichever document is larger.

### Guardrails

| Guardrail | Where | What it catches |
|---|---|---|
| Off-topic scope check | System prompt | Questions unrelated to career fit or uploaded docs |
| Empty question | `chain.py` input validation | Blank or whitespace-only input |
| Question length limit | `chain.py` input validation | Questions over 1000 characters |
| Empty PDF | `parser.py` | Image-only PDFs with no extractable text |
| No documents uploaded | `chain.py` retriever check | Questions asked before any files are ingested |
| Low-confidence signal | LLM instruction | Instructs LLM to set confidence < 0.4 when context is thin |

What's not implemented (would add in production): rate limiting per user, PII redaction before embedding, output filtering for harmful content.

### Structured Output (No Hand-Parsed JSON)
The chain uses `.with_structured_output(CareerAnswer)` which constrains the LLM to emit valid JSON matching the Pydantic schema. This means:
- No `json.loads()` anywhere in the codebase
- No `KeyError` at runtime if the LLM omits a field
- Pydantic validates confidence range `[0.0, 1.0]` and skill importance literals at the type level

### Confidence Scoring
The LLM is prompted to set `confidence` based on how well the retrieved context supports the answer — not just how certain it is in general. Low-context queries produce low-confidence answers rather than confident-but-hallucinated ones.

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| UI framework | Streamlit | 2-day deadline. Solid engineering > flashy UI. |
| LLM | gpt-4o-mini | Best cost/quality for structured output tasks |
| Embeddings | text-embedding-3-small | Strong semantic search, low cost, 1536 dims |
| Vector DB | ChromaDB | Persists to disk — no re-embedding on restart |
| Structured output | `.with_structured_output(CareerAnswer)` | Eliminates hand-parsing, type-safe at runtime |
| Two collections | cv + jd separate | Enables scoped retrieval and cross-collection merge |
| Score assignment | Rank-based (1.0 → 0.5 decay) | ChromaDB L2 scores are negative and unreliable for threshold filtering |

---

## What Would Be Required to Productionise on AWS

### Compute
- Containerise with Docker (already done), deploy to **AWS ECS Fargate** — no EC2 management, scales to zero between uses.
- Alternatively: **AWS App Runner** for simpler Streamlit deployment.

### Vector Store
- Replace local ChromaDB with **Amazon OpenSearch Serverless** (vector engine) or **Pinecone** for multi-user isolation, horizontal scaling, and managed backups.
- One collection (namespace) per user session, keyed by user ID.

### Authentication
- **AWS Cognito** user pool in front of the app — each user's documents stay private via namespace-scoped collection keys.

### Document Storage
- Uploaded PDFs stored in **S3** (not tempfiles) — durable, auditable, enables async re-processing.
- S3 event triggers **Lambda** for background ingestion to avoid blocking the UI.

### LLM and Embedding
- OpenAI API calls made from **Lambda or ECS** — keep API keys in **AWS Secrets Manager**, not `.env`.
- Rate limiting and retry logic via **tenacity** or a queue (SQS) for burst traffic.

### Observability
- LangSmith already integrated for LLM tracing.
- Add **CloudWatch** metrics for latency, error rate, and per-user token spend.
- Set cost alerts on OpenAI usage.

### CI/CD
- **GitHub Actions** pipeline: `pytest` on PR, docker build + push to ECR on merge to main, ECS deploy.

---

## Engineering Standards

### Followed
- Pydantic structured output — no hand-parsed JSON
- Type annotations throughout
- Failure paths tested (empty documents, below-threshold retrieval, LLM errors, empty questions)
- Semantic similarity evals, not keyword matching
- Docstrings on all public functions explaining the *why*, not the *what*
- ChromaDB persistence — no re-embedding on restart
- `ValueError` raised with clear messages on invalid inputs (empty PDF, empty text, empty question)

### Skipped (and why)
- **FastAPI backend**: Streamlit handles the UI layer — adding FastAPI would introduce a deployment unit with no benefit for a single-user take-home app. In production, I'd add it for multi-user API access.
- **Authentication**: Out of scope for the take-home. Would add Cognito in production.
- **Async ingestion**: File ingestion is synchronous. For production, large PDFs would be queued via SQS + Lambda.
- **Reranking (cross-encoder)**: Would improve retrieval quality for ambiguous queries. Left out to keep complexity in scope.

---

## Screenshots

**Main answer view** — 80% confidence, sources cited, structured response:

![Main answer](screenshots/01-main-answer.png)

**Skill gaps** — critical gaps (🔴) and nice-to-have gaps (🟡) with suggestions:

![Skill gaps](screenshots/02-skill-gaps.png)

**Follow-up questions** — suggested next questions based on the answer:

![Follow-up questions](screenshots/03-follow-up-questions.png)

---

## How AI Tools Were Used

I used Claude throughout the build, mainly for boilerplate: Pydantic schemas, test file scaffolding, and a first draft of the system prompt.

Where I let it generate freely: repetitive structure that I'd have to write anyway (dataclass definitions, pytest fixtures, the Makefile). Fast to review, low risk of subtle bugs.

Where I didn't use it: architectural decisions, debugging, and the eval dataset. The two-collection ChromaDB design, rank-based scoring, and structured output approach were things I reasoned through myself. The ChromaDB negative score bug was something I had to find and fix by reading the retriever output. The eval dataset I wrote case by case so I understood exactly what each one was testing.

How I kept it maintainable: I reviewed every generated file before committing. If generated code had a pattern I didn't understand or wouldn't write myself, I rewrote it. The rule I followed was: if I couldn't explain every line in a code review, I didn't ship it.

What worked well: using it for boilerplate, asking it to suggest edge cases for tests, getting a first draft of config and CI files fast.

What I avoided: using it for anything involving data flow decisions, trusting generated error handling (it tends to swallow exceptions), and letting it write tests without checking the assertions actually tested the right thing.

---

## What I'd Do Differently With More Time

1. **Reranking**: Add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation to improve chunk selection for ambiguous queries.
2. **Streaming responses**: Use `ChatOpenAI` with `streaming=True` and Streamlit's `st.write_stream` for a more responsive feel.
3. **Multi-turn memory**: Add `ConversationBufferMemory` so follow-up questions ("what about the second point?") resolve correctly.
4. **Async ingestion**: Move PDF processing to a background thread so large CVs don't block the UI.
5. **Per-user sessions**: In multi-user deployment, scope ChromaDB collections by session ID (UUID) and clean up on session end.
6. **Eval automation**: Run evals in CI against golden answers after each model upgrade or prompt change.
7. **Metadata filtering**: Add ChromaDB `where` filters so "questions about Job #2" only retrieve chunks from that JD file.
