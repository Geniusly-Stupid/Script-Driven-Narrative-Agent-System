# Script-Driven Narrative Agent System

Production-oriented interactive storytelling platform with strict step-by-step progression, LangGraph orchestration, MongoDB persistence, and Chroma-based RAG.

## Overview

This project implements a script-driven narrative engine that:

1. Uploads and parses PDF scripts
2. Structures scripts into Scene -> Plot hierarchy
3. Stores world + runtime state in MongoDB
4. Performs semantic retrieval with Chroma (not MongoDB semantic search)
5. Runs a LangGraph narrative state machine
6. Exposes a polished React + Tailwind step-guided UI

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

Indexes are created on startup (scene IDs, plot/scene summary IDs, conversation keys, etc.).

## Sequential Workflow Enforcement

Backend enforces strict stage progression using `system_state`:

1. `POST /api/workflow/upload-script` only at stage `upload`
2. `POST /api/workflow/confirm-structure` only at stage `parse`
3. `POST /api/workflow/character` only at stage `character`
4. `POST /api/workflow/session/message` only at stage `session`

Invalid stage transitions return HTTP 409.

## Chroma Vector RAG

Knowledge docs are created from parsed script entities and stored in Chroma with metadata:

- `scene_id`
- `plot_id`
- `type`
- `name`

Dynamic retrieval sequence per turn:

1. Generate retrieval queries from user input + current goal + memory
2. Embed query
3. Retrieve top-k similar documents from Chroma
4. Inject into narrative prompt context

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

Interaction follows strict ordered stages (prompt -> context -> response -> memory -> plot check -> scene check).

## Prompt Template

The prompt is built with fixed sections:

- `SYSTEM`
- `USER INPUT`
- `SCRIPT STATE`
- `MEMORY`
- `RETRIEVED KNOWLEDGE`
- `INTERNAL STATE`
- `INSTRUCTION`

Dice tool supported: `roll_dice(dice_type)` with enforced non-fabrication behavior and explicit result injection.

## Setup

## 1. Backend

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy backend\.env.example backend\.env
uvicorn backend.main:app --reload --port 8000
```

## 2. Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Frontend default URL: `http://localhost:5173`
Backend default URL: `http://localhost:8000`

## Environment Variables

Backend (`backend/.env`):

- `MONGODB_URI`
- `MONGODB_DB`
- `CHROMA_PATH`
- `OPENAI_API_KEY` (optional; fallback parser/response behavior works without it)
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

## Future Improvements

1. Replace heuristic fallback parser with robust structured extraction prompt + schema validation retries.
2. Add auth + multi-user campaign namespaces.
3. Add observability (OpenTelemetry, tracing per LangGraph node).
4. Add richer dice orchestration through explicit tool-calling JSON schema.
5. Add integration tests for stage transitions and progression logic.
