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

## Notes

- This project is designed for local standalone execution.
- All state is persisted locally (`narrative.db` and `.chroma/`).
- No API server is required.
