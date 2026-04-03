# Script-Driven Narrative Agent System

A local, standalone AI game master designed for narrative role-playing games. It manages long-context interactive storytelling based on structured scripts, while allowing players to freely explore the story. The system runs as a single Python application and focuses on maintaining coherent plots, consistent characters, and immersive gameplay.

## Project Overview

### Background

Tabletop role-playing games (TRPGs), such as Call of Cthulhu (CoC) and Dungeons & Dragons (DnD), are popular worldwide. However, traditional gameplay requires a human game master, which limits accessibility. The game master must prepare in advance and may not respond in real time. Accessing scripts in unfamiliar languages is also difficult. An AI game master can reduce these barriers by offering a more accessible, responsive, and multilingual storytelling experience.

### Problem Statement

TRPG scripts are often long, sometimes spanning hundreds of pages, and ongoing player interactions further extend the context. This makes it difficult for standard LLM-based systems to maintain long-term narrative coherence. Common issues include plot drift and inconsistencies in characters and facts.

### Goals

The main goal of the project is to design a script-driven AI game master that manages long-context interactive storytelling for narrative role-playing games. The system aims to:

- Ensure stable plot progression
- Maintain consistency in characters and facts
- Support player freedom across story branches

### Technical Highlights

The current implementation reflects these goals through:

- structured scripts organized into scenes and plots
- plot-level progression control through explicit state updates and completion checks
- dynamic context construction using memory lookup, retrieval, and response generation
- separated dice-check detection, deterministic rolling, and final narrative generation
- CoC rule knowledge ingestion from `database/GameRules.md`
- multilingual output control in the UI while keeping parsing language-agnostic


## Architecture

```text
+------------------------- Streamlit UI --------------------------+
| Upload Script -> Review Parse -> Create Character -> Chat Loop |
+-------------------------------+---------------------------------+
                                |
                                v
+-------------------- LangGraph Narrative Agent -------------------+
| build_prompt -> retrieve_memory -> generate_retrieval_queries    |
| -> vector_retrieve -> construct_context                          |
| -> check_whether_roll_dice -> (roll_dice or pass)               |
| -> generate_response -> write_memory                            |
| -> check_plot_completion -> check_scene_completion -> update_state|
+-------------------------------+----------------------------------+
                                |
                 +--------------+--------------+
                 v                             v
            SQLite (structured)          Chroma (semantic)
      scenes/plots/memory/summaries      npc/location/event/rule docs
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
    rules_loader.py
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
- `knowledge_base`

Usage in runtime:
- script parse result persistence
- external rule knowledge persistence
- current scene/plot pointer tracking
- turn memory append/read
- plot/scene summaries
- strict stage lifecycle (`upload -> parse -> character -> session`)
- output language persistence

### `app/vector_store.py` (Chroma)
Responsible only for semantic retrieval.

- deterministic local embedding function
- ingestion from scene/plot entities (NPC, location, events)
- ingestion from knowledge entries, including CoC rules
- top-k similarity search for context enrichment

### `app/parser.py`
Script ingestion logic.

- reads PDF pages
- segments pages into scenes (chunk-based)
- segments scenes into plots (marker/heuristic)
- extracts goals and entities used by state + RAG

The parser remains language-agnostic and follows the uploaded script content rather than the UI output language setting.

### `app/agent_graph.py`
LangGraph orchestration engine for retrieval, dice checks, deterministic roll resolution, and narrative progression.

Nodes:
1. `build_prompt`
2. `retrieve_memory`
3. `generate_retrieval_queries`
4. `vector_retrieve`
5. `construct_context`
6. `check_whether_roll_dice`
7. `roll_dice`
8. `generate_response`
9. `write_memory`
10. `check_plot_completion`
11. `check_scene_completion`
12. `update_state`

Storage calls are adapted to SQLite, retrieval calls to Chroma, and dice success evaluation is handled deterministically outside the LLM.

### `app/rag.py`
RAG helper pipeline:
- generate retrieval queries from user input + plot goal + events + memory tail
- classify retrieved docs into prompt sections

Retrieved knowledge includes both script-derived knowledge and rule documents.

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

The UI also includes:
- output language selection (`English` / `Chinese`)
- debug prompt inspection
- CoC-style assisted character creation
- visible dice roll and skill check results during play

## Narrative Turn Execution

For each chat message:

`User input`
-> `build_prompt`
-> `retrieve_memory` (SQLite)
-> `generate_retrieval_queries`
-> `vector_retrieve` (Chroma)
-> `construct_context`
-> `check_whether_roll_dice`
-> `roll_dice` (only when a check is required)
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
  - output language
  - persistent knowledge entries

- **Chroma**: semantic similarity only.
  - vectorized knowledge docs
  - retrieval for context augmentation
  - retrieval over world knowledge and CoC rule knowledge

This separation keeps deterministic state operations isolated from semantic search operations.

## Setup

Before running the project, get an API key from:

[https://build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)

Then create a file named:

```bash
api_key.txt
```

Place it in the project root directory and paste your API key inside (only the key, no extra spaces, utf-8 encoding).

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

## Notes

- This project is designed for local standalone execution.
- All state is persisted locally (`narrative.db` and `.chroma/`).
- `database/GameRules.md` is loaded into the knowledge base as Call of Cthulhu rules knowledge.

## Debug Scripts

A new `test/` directory has been added.
These scripts all use **mock inputs + print outputs**, allowing you to manually verify whether the results match expectations.

Directory contents (each file except `app/ui.py` has a corresponding test):

* `test/test_llm_generate.py`: Tests the raw LLM API call (streaming behavior, thinking/reasoning/content parsing).
  
  ⚠️ **Before running this test, make sure `api_key.txt` has been created and configured as described in the Setup section.**

  ⚠️ **All LLM invocation logic in the project must stay aligned with this file.**

  Any changes to request payload, headers, or streaming parsing should be validated here first.
* `test/test_init.py`: Tests importing `app/__init__.py`
* `test/test_database.py`: Tests `app/database.py` (database creation, insert, read, state updates)
* `test/test_parser.py`: Tests `app/parser.py` (mock page parsing for Scene/Plot)
* `test/test_rag.py`: Tests `app/rag.py` (query generation and knowledge classification)
* `test/test_state.py`: Tests `app/state.py` (plot/scene progression and transitions)
* `test/test_vector_store.py`: Tests `app/vector_store.py` (insertion and retrieval)
* `test/test_rules_loader.py`: Tests rule chunking + vector insertion + retrieval from `GameRules.md`
* `test/test_roll_workflow.py`: Tests the separated dice-check workflow and deterministic success evaluation
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

