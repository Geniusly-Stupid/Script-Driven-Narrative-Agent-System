# Script-Driven Narrative Agent System

Production-oriented interactive storytelling platform with strict stage progression, LangGraph orchestration, MongoDB persistence, and Chroma-based retrieval.

## Overview

This project provides a script-driven narrative engine that:

1. Ingests PDF scripts.
2. Structures script content into Scene -> Plot hierarchy.
3. Stores authoritative runtime data in MongoDB.
4. Uses Chroma for semantic retrieval (RAG).
5. Runs a LangGraph-based narrative agent for each user turn.
6. Exposes a React + Tailwind step-based UI (upload -> parse -> character -> session).

## Architecture

```text
+-------------------- Frontend (React + Tailwind) --------------------+
| Step 1 Upload | Step 2 Parse Review | Step 3 Character | Step 4 Chat |
+-------------------------------+---------------------------------------+
                                |
                                v
+------------------------- FastAPI Backend ----------------------------+
| Routers (workflow) -> Services -> Agent -> Repository -> MongoDB     |
|                                   |                                   |
|                                   +-> Chroma Vector Store            |
+---------------------------------------------------------------------+
```

## Backend Structure

```text
/backend
  /agents
    narrative_graph.py
  /database
    mongo.py
    repository.py
  /models
    schemas.py
  /routers
    workflow.py
  /services
    llm_service.py
    pdf_parser.py
    script_parser.py
  /vector_store
    chroma_store.py
  config.py
  main.py
```

Backend is intentionally split into:
- `routers`: HTTP entrypoints and workflow gating.
- `services`: ingestion/parsing/LLM-facing logic.
- `agents`: LangGraph runtime orchestration.
- `database`: Mongo connection + repository operations.
- `vector_store`: Chroma retrieval layer.

## Frontend Structure

```text
/frontend
  /src
    /api
    /components
    /hooks
    /types
    App.tsx
    main.tsx
    index.css
```

Frontend remains thin:
- step-based progression UI,
- chat interface for narrative turns,
- side status panel for scene/plot/progress.

## MongoDB Schema

### `scenes`
- `scene_id`, `scene_goal`, `plots[]`, `status`, `scene_summary`
- Plot fields: `plot_id`, `plot_goal`, `mandatory_events`, `npc`, `locations`, `status`, `progress`

### `knowledge_base`
- `type`: `npc | location | item | event | rule`
- `name`, `description`, `metadata`

### `player_profiles`
- `name`, `background`, `traits`, `stats`, `special_skills`

### `conversation_memory`
- `scene_id`, `plot_id`, `turns[]` (`user`, `agent`, `timestamp`)

### `plot_summaries`
- plot-level completion summaries

### `scene_summaries`
- scene-level completion summaries

### `system_state`
- `current_scene_id`, `current_plot_id`, `plot_progress`, `scene_progress`, `stage`
- stage enum: `upload | parse | character | session`

Indexes are created on startup for scene IDs, knowledge lookup dimensions, conversation keys, summaries, and stage access paths.

## Sequential Workflow Enforcement

Backend enforces strict stage progression using `system_state`:

1. `POST /api/workflow/upload-script` only at stage `upload`
2. `POST /api/workflow/confirm-structure` only at stage `parse`
3. `POST /api/workflow/character` only at stage `character`
4. `POST /api/workflow/session/message` only at stage `session`

Invalid transitions return HTTP 409.

This strict gating prevents:
- session turns before script parse,
- character creation after session starts,
- accidental state corruption from out-of-order operations.

## Chroma Vector RAG

Knowledge docs are generated from parsed script entities and stored in Chroma with metadata:
- `scene_id`
- `plot_id`
- `type`
- `name`

Dynamic retrieval sequence per turn:
1. Generate retrieval queries from user input + plot context + memory tail.
2. Embed/query against Chroma.
3. Collect top-k semantic matches.
4. Insert retrieved facts into prompt context.

Why retrieval is separated from Mongo:
- MongoDB remains source-of-truth for transactional/runtime state.
- Chroma is optimized for vector similarity search.
- Separation avoids mixing exact-state updates with semantic ranking concerns.

## LangGraph Flow

Nodes:
1. `build_prompt`
2. `retrieve_memory`
3. `generate_retrieval_queries`
4. `vector_retrieve`
5. `construct_context`
6. `generate_response`
7. `write_memory`
8. `check_plot_completion`
9. `check_scene_completion`
10. `update_state`

Execution order is fixed by graph edges, so every turn follows the same deterministic stage sequence.

## Narrative Agent Internal Execution Flow

This is the core runtime path for one user message.

### State object design

The LangGraph state carries both control and content fields, including:
- Routing/runtime: `scene_id`, `plot_id`, `plot_progress`, `scene_progress`
- User/session: `latest_user_input`, `player_profile`
- Memory/RAG: `conversation_history`, `retrieved_docs`, `retrieval_queries`
- Prompt assembly: `system_prompt`, `user_prompt`, `context`, `prompt`
- Script constraints: `scene_goal`, `plot_goal`, `mandatory_events`
- Summaries: `previous_plot_summary`, `current_scene_summary`
- Outputs/control flags: `response`, `dice_result`, `plot_completed`, `scene_completed`

### Node responsibilities and data flow

- `build_prompt`
  - Initializes prompt primitives (`system_prompt`, `user_prompt`).

- `retrieve_memory`
  - Pulls recent conversation turns for current `(scene_id, plot_id)`.
  - Loads current scene document to inject script constraints:
    - scene goal,
    - current plot goal,
    - mandatory events,
    - current scene summary.
  - Loads previous plot summary if prior plot exists.

- `generate_retrieval_queries`
  - Builds retrieval queries from:
    - latest user input,
    - real current plot goal,
    - mandatory events,
    - recent user-message tail.

- `vector_retrieve`
  - Executes Chroma semantic search per query.
  - Stores condensed top results in `retrieved_docs`.

- `construct_context`
  - Constructs complete prompt from the fixed template sections:
    - SYSTEM PROMPT
    - USER INPUT
    - SCRIPT STATE
    - MEMORY
    - RETRIEVED KNOWLEDGE
    - INTERNAL STATE
    - INSTRUCTION
  - Injects memory and retrieved facts into named placeholders.

- `generate_response`
  - Detects dice intent.
  - Executes dice tool when needed.
  - Calls LLM with fully assembled prompt.
  - Stores model output in `response`.

- `write_memory`
  - Appends `{user, agent, timestamp}` into `conversation_memory`
    keyed by current `scene_id` and `plot_id`.

- `check_plot_completion`
  - Evaluates progress via LLM JSON evaluator (with fallback defaults).
  - Updates `plot_progress`.
  - On completion, writes plot summary to `plot_summaries`.

- `check_scene_completion`
  - Marks current plot completed in scene document.
  - Recomputes scene progress from completed plots.
  - On scene completion, generates and stores scene summary.

- `update_state`
  - Advances to next plot or next scene when appropriate.
  - Generates scene introduction when entering next scene.
  - Persists canonical runtime status to `system_state`.

### One-turn execution path (step-by-step)

User input  
-> `build_prompt`  
-> `retrieve_memory`  
-> `generate_retrieval_queries`  
-> `vector_retrieve`  
-> `construct_context`  
-> `generate_response`  
-> `write_memory`  
-> `check_plot_completion`  
-> `check_scene_completion`  
-> `update_state`

### Simplified flow diagram

```text
User Message
   |
   v
[Load state + memory + script constraints]
   |
   v
[Generate RAG queries] -> [Chroma retrieval]
   |                          |
   +------------> [Context Assembly]
                            |
                            v
                        [LLM Response]
                            |
                            v
                      [Write turn memory]
                            |
                            v
                 [Plot completion evaluation]
                            |
                            v
                [Scene completion evaluation]
                            |
                            v
                [Persist next system_state]
```

### Prompt construction details

Prompt is assembled from a fixed template with strict section order and stable placeholders.  
This guarantees predictable prompting for debugging and consistent model behavior across turns.

### Progression and strict ordering

Two independent layers enforce order:

1. **Graph ordering**: LangGraph edges guarantee node sequence within each turn.
2. **Workflow ordering**: API stage checks in `system_state` guarantee user cannot skip lifecycle stages.

### Fallback logic

Fallback behavior is explicit and localized:
- Script ingestion: heuristic parser fallback if structured LLM parse is unavailable/invalid.
- LLM completion: fallback text/json outputs when model is unavailable.
- Completion checks: default progress increments when evaluator output is missing/invalid.

This keeps the system operational in degraded environments while preserving flow integrity.

## Prompt Template

The runtime uses a fixed structured template with these top-level sections:

- `SYSTEM PROMPT`
- `USER INPUT`
- `SCRIPT STATE`
- `MEMORY`
- `RETRIEVED KNOWLEDGE`
- `INTERNAL STATE`
- `INSTRUCTION`

Dice/tool instructions are embedded in template constraints and interpreted during response generation.

## Setup

### 1. Backend

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy backend\.env.example backend\.env
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Frontend: `http://localhost:5173`  
Backend: `http://localhost:8000`

## Environment Variables

Backend (`backend/.env`):
- `MONGODB_URI`
- `MONGODB_DB`
- `CHROMA_PATH`
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL`

Frontend (`frontend/.env`):
- `VITE_API_BASE`

## API Summary

- `GET /health`
- `GET /api/workflow/status`
- `POST /api/workflow/upload-script`
- `GET /api/workflow/scenes`
- `POST /api/workflow/confirm-structure`
- `POST /api/workflow/character`
- `POST /api/workflow/session/message`

## Debugging Without Frontend

Use standalone scripts in `/tests` to validate backend components without running FastAPI or frontend.

1. Test MongoDB connection/indexes/CRUD cleanup:

```bash
python tests/test_database.py
```

2. Test Chroma add/query/reset:

```bash
python tests/test_vector_store.py
```

3. Test script parsing + page tracking:

```bash
python tests/test_script_parser.py
```

4. Test one LangGraph turn (prompt/retrieval/response/state):

```bash
python tests/test_agent_graph.py
```

5. Test RAG query generation + retrieval:

```bash
python tests/test_rag_pipeline.py
```

6. Run all tests with summary:

```bash
python tests/run_all_tests.py
```

## Future Improvements

1. Replace heuristic fallback parser with stronger schema-validated extraction retries.
2. Add auth and campaign-level multi-tenant separation.
3. Add tracing/observability for node-by-node runtime diagnostics.
4. Expand structured tool-calling contracts for mechanics-heavy systems.
5. Add deeper automated coverage for stage transitions and progression edge cases.