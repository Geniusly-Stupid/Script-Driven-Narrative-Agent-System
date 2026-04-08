# Script-Driven Narrative Agent System

A local standalone script-driven narrative agent built as a single Python app.

- **LangGraph** orchestrates turn-by-turn narrative execution.
- **SQLite** stores structured state (scenes, plots, memory, summaries, system state).
- **Chroma** handles semantic retrieval (RAG) for NPC/location/event knowledge.
- **Streamlit** provides the interactive UI.


## Architecture

```text
+------------------------- Streamlit UI --------------------------+
| Upload Script -> Review Parse -> Create Character -> Chat Loop |
+-------------------------------+---------------------------------+
                                |
                                v
+-------------------- LangGraph Narrative Agent -------------------+
| build_prompt -> retrieve_memory -> generate_retrieval_queries    |
| -> vector_retrieve -> construct_context -> generate_response     |
| -> write_memory -> check_plot_completion -> check_scene_completion|
| -> update_state                                                   |
+-------------------------------+----------------------------------+
                                |
                 +--------------+--------------+
                 v                             v
            SQLite (structured)          Chroma (semantic)
      scenes/plots/memory/summaries      npc/location/event docs
               /system_state             + similarity retrieval
```

## Project Structure

```text
/app
    database.py
    vector_store.py
    parser.py
    agent_graph.py
    state.py
    rag.py
    ui.py

main.py
requirements.txt
README.md
llm_backend.txt          # qwen | openai (which LLM backend to use)
```

## Core Components

### `app/database.py` (SQLite)
Responsible only for structured persistence.

Tables:
- `scenes`
- `plots`
- `memory`
- `summaries`
- `system_state`

Usage in runtime:
- script parse result persistence
- current scene/plot pointer tracking
- turn memory append/read
- plot/scene summaries
- strict stage lifecycle (`upload -> parse -> character -> session`)

### `app/vector_store.py` (Chroma)
Responsible only for semantic retrieval.

- deterministic local embedding function
- ingestion from scene/plot entities (NPC, location, events)
- top-k similarity search for context enrichment

### `app/parser.py`
Script ingestion logic.

- reads PDF pages
- segments pages into scenes (chunk-based)
- segments scenes into plots (marker/heuristic)
- extracts goals and entities used by state + RAG

### `app/agent_graph.py`
LangGraph orchestration engine (preserved node flow).

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

Storage calls are adapted to SQLite, retrieval calls to Chroma. Node sequence and responsibilities remain intact.

### `app/rag.py`
RAG helper pipeline:
- generate retrieval queries from user input + plot goal + events + memory tail
- classify retrieved docs into prompt sections

### `app/state.py`
Progression helpers:
- plot completion evaluation
- scene completion evaluation
- next scene/plot transition calculation

### `app/ui.py` + `main.py`
Streamlit runtime.

Flow:
1. Upload PDF
2. Parse and review scene structure
3. Create character
4. Start narrative chat session
5. Display scene/plot/progress and retrieved knowledge

## Narrative Turn Execution

For each chat message:

`User input`
-> `build_prompt`
-> `retrieve_memory` (SQLite)
-> `generate_retrieval_queries`
-> `vector_retrieve` (Chroma)
-> `construct_context`
-> `generate_response`
-> `write_memory` (SQLite)
-> `check_plot_completion`
-> `check_scene_completion`
-> `update_state` (SQLite)

## SQLite vs Chroma Responsibilities

- **SQLite**: canonical structured runtime state.
  - scene and plot status/progress
  - chat memory turns
  - summaries
  - global system stage and active pointers

- **Chroma**: semantic similarity only.
  - vectorized knowledge docs
  - retrieval for context augmentation

This separation keeps deterministic state operations isolated from semantic search operations.

## Setup

### 1) Choose the LLM backend (one file, default is Qwen / NVIDIA)

In the project root, edit **`llm_backend.txt`**. The first non-empty line that is not a comment (`#`) must start with either:

- **`qwen`** — use the NVIDIA Build API with Qwen models (same as writing `nvidia`; this is the **default**).
- **`openai`** — use the OpenAI Chat Completions API instead.

Example:

```text
qwen
```

You can optionally specify a model on the same line:

```text
qwen qwen/qwen3.5-397b-a17b
```

```text
openai gpt-4o-mini
```

To switch to OpenAI, change that line to:

```text
openai
```

Reference copy: **`llm_backend.example.txt`** (you can duplicate it to `llm_backend.txt` if needed).

**Override (optional):** for CI or one-off runs, set environment variable `LLM_PROVIDER` to `qwen`, `nvidia`, or `openai`. It takes precedence over `llm_backend.txt`.

### 2) API keys

**Qwen / NVIDIA** (when backend is `qwen` or `nvidia`):

- Get a key from [NVIDIA Build API keys](https://build.nvidia.com/settings/api-keys).
- Put it in project root **`api_key.txt`** (single line), or set **`NVIDIA_API_KEY`**, or use a combined file (see below).

**OpenAI** (when backend is `openai`):

- Set **`OPENAI_API_KEY`**, or put the key in **`openai_api_key.txt`** (one line, bare `sk-...`), or add a line to **`api_key.txt`**:

```text
OPENAI_API_KEY=sk-...
```

If more than one source is set, **only one is used** (no merge): **`OPENAI_API_KEY` env → `openai_api_key.txt` → `api_key.txt`** with `OPENAI_API_KEY=...`. Same key in two places does not cause errors; the first in that order wins.

**Both backends in one `api_key.txt` (optional):**

```text
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...
```

OpenAI-only tuning (optional):

```bash
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_MAX_TOKENS=2048
export OPENAI_TEMPERATURE=0.6
export OPENAI_TOP_P=0.95
```

No code changes are required to switch: edit **`llm_backend.txt`** and ensure the matching API key is configured.

Once the API key is set up, create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
streamlit run main.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Context Compression

To keep prompts within a safe context window, the runtime automatically compresses context when needed.

- Trigger condition: in `app/agent_graph.py`, `context_compressed` becomes `True` when rendered prompt length exceeds `CONTEXT_MAX_CHARS`.
- Default threshold: `CONTEXT_MAX_CHARS=18000`
- Recent full-turn window: `CONTEXT_RECENT_TURNS=3` (latest 2-3 turns are included in context before compression fallback)

Set environment variables before running:

```bash
export CONTEXT_MAX_CHARS=6000
export CONTEXT_RECENT_TURNS=3
streamlit run main.py
```

In UI debug mode, each turn shows:
- `context_compressed`
- `raw_chars` (before compression)
- `final_chars` (after compression)

This makes it easy to verify whether compression is activated and how much context was reduced.

## Notes

- This project is designed for local standalone execution.
- All state is persisted locally (`narrative.db` and `.chroma/`).

## Debug Scripts

A new `test/` directory has been added.
These scripts all use **mock inputs + print outputs**, allowing you to manually verify whether the results match expectations.

Directory contents (each file except `app/ui.py` has a corresponding test):

* `test/test_llm_generate.py`: Tests the raw LLM API call (streaming behavior, thinking/reasoning/content parsing).
  
  ⚠️ **Before running this test, make sure API keys are configured as described in the Setup section** (`api_key.txt` / env vars).

  ⚠️ **All LLM invocation logic in the project must stay aligned with this file.**

  Any changes to request payload, headers, or streaming parsing should be validated here first.
* `test/test_init.py`: Tests importing `app/__init__.py`
* `test/test_database.py`: Tests `app/database.py` (database creation, insert, read, state updates)
* `test/test_parser.py`: Tests `app/parser.py` (mock page parsing for Scene/Plot)
* `test/test_rag.py`: Tests `app/rag.py` (query generation and knowledge classification)
* `test/test_state.py`: Tests `app/state.py` (plot/scene progression and transitions)
* `test/test_vector_store.py`: Tests `app/vector_store.py` (insertion and retrieval)
* `test/test_agent_graph.py`: Tests `app/agent_graph.py` (complete single-turn workflow)
* `test/test_main.py`: Tests `main.py` import and entry-point availability
* `test/run_all.py`: Executes all test scripts sequentially

Run individually:

```bash
python test/test_llm_generate.py
python test/test_database.py
python test/test_parser.py
python test/test_agent_graph.py
```

Run all at once:

```bash
python test/run_all.py
```
