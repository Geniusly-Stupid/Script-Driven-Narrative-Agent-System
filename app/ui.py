from __future__ import annotations

import json
import random
import re

import streamlit as st

from app.agent_graph import NarrativeAgent
from app.database import Database
from app.parser import parse_script_bundle, read_pdf_pages
from app.rules_loader import load_game_rules_knowledge
from app.vector_store import ChromaStore


COC_CORE_KEYS = ['STR', 'CON', 'SIZ', 'DEX', 'APP', 'INT', 'POW', 'EDU']
COC_3D6_KEYS = ['STR', 'CON', 'DEX', 'APP', 'POW']
COC_2D6_KEYS = ['SIZ', 'INT', 'EDU']
COC_ARCHETYPES = [
    {
        'name': 'Detective',
        'weights': {'INT': 3.0, 'DEX': 2.0, 'POW': 2.0, 'APP': 1.0, 'EDU': 2.0},
        'occupation_skills': [('Spot Hidden', 22), ('Psychology', 16), ('Law', 14), ('Listen', 14), ('Stealth', 12), ('Firearms', 12), ('Persuade', 10)],
        'interest_skills': [('Library Use', 30), ('Occult', 25), ('Locksmith', 25), ('First Aid', 20)],
    },
    {
        'name': 'Scholar',
        'weights': {'EDU': 3.0, 'INT': 3.0, 'POW': 1.5, 'APP': 1.0, 'DEX': 0.5},
        'occupation_skills': [('Library Use', 24), ('History', 18), ('Archaeology', 14), ('Language (Other)', 14), ('Anthropology', 12), ('Occult', 10), ('Persuade', 8)],
        'interest_skills': [('Spot Hidden', 30), ('Psychology', 25), ('Credit Rating', 25), ('Charm', 20)],
    },
    {
        'name': 'Journalist',
        'weights': {'INT': 2.5, 'APP': 2.0, 'POW': 2.0, 'DEX': 1.5, 'EDU': 1.5},
        'occupation_skills': [('Persuade', 22), ('Fast Talk', 18), ('Library Use', 16), ('Psychology', 14), ('Photography', 12), ('Spot Hidden', 10), ('Stealth', 8)],
        'interest_skills': [('Credit Rating', 30), ('Occult', 25), ('Listen', 25), ('Drive Auto', 20)],
    },
    {
        'name': 'Soldier',
        'weights': {'STR': 2.5, 'CON': 2.5, 'DEX': 2.0, 'POW': 1.5, 'SIZ': 1.5},
        'occupation_skills': [('Firearms', 24), ('Fighting (Brawl)', 20), ('Dodge', 14), ('Survival', 12), ('First Aid', 12), ('Navigate', 10), ('Intimidate', 8)],
        'interest_skills': [('Spot Hidden', 30), ('Listen', 25), ('Mechanical Repair', 25), ('Psychology', 20)],
    },
    {
        'name': 'Doctor',
        'weights': {'EDU': 3.0, 'INT': 2.5, 'DEX': 1.5, 'POW': 1.5, 'APP': 1.0},
        'occupation_skills': [('Medicine', 26), ('First Aid', 20), ('Science (Biology)', 16), ('Psychology', 12), ('Pharmacy', 10), ('Persuade', 8), ('Listen', 8)],
        'interest_skills': [('Library Use', 30), ('Spot Hidden', 25), ('Occult', 25), ('Drive Auto', 20)],
    },
    {
        'name': 'Explorer',
        'weights': {'CON': 2.0, 'SIZ': 2.0, 'DEX': 2.0, 'POW': 1.5, 'INT': 1.5},
        'occupation_skills': [('Survival', 24), ('Navigate', 18), ('Climb', 14), ('Track', 14), ('Spot Hidden', 12), ('First Aid', 10), ('Natural World', 8)],
        'interest_skills': [('Firearms', 30), ('Mechanic Repair', 25), ('Listen', 25), ('Anthropology', 20)],
    },
    {
        'name': 'Antiquarian',
        'weights': {'EDU': 2.5, 'APP': 2.0, 'INT': 2.0, 'POW': 1.5, 'DEX': 1.0},
        'occupation_skills': [('Appraise', 22), ('History', 18), ('Charm', 14), ('Persuade', 14), ('Library Use', 12), ('Credit Rating', 10), ('Occult', 10)],
        'interest_skills': [('Spot Hidden', 30), ('Psychology', 25), ('Stealth', 25), ('Language (Other)', 20)],
    },
    {
        'name': 'Professor',
        'weights': {'EDU': 3.0, 'INT': 2.5, 'APP': 1.0, 'POW': 1.5, 'CON': 1.0},
        'occupation_skills': [('Library Use', 24), ('Language (Own)', 16), ('Language (Other)', 16), ('Psychology', 12), ('History', 12), ('Persuade', 10), ('Occult', 10)],
        'interest_skills': [('Spot Hidden', 30), ('Credit Rating', 25), ('Charm', 25), ('Listen', 20)],
    },
    {
        'name': 'Private Eye',
        'weights': {'INT': 2.5, 'DEX': 2.0, 'STR': 1.5, 'POW': 2.0, 'APP': 1.0},
        'occupation_skills': [('Spot Hidden', 22), ('Stealth', 16), ('Locksmith', 14), ('Psychology', 14), ('Firearms', 12), ('Dodge', 12), ('Law', 10)],
        'interest_skills': [('Listen', 30), ('Drive Auto', 25), ('Occult', 25), ('Persuade', 20)],
    },
    {
        'name': 'Artist',
        'weights': {'APP': 2.5, 'POW': 2.0, 'DEX': 2.0, 'INT': 1.5, 'EDU': 1.0},
        'occupation_skills': [('Art/Craft', 24), ('Psychology', 16), ('Charm', 16), ('Persuade', 14), ('History', 10), ('Spot Hidden', 10), ('Listen', 10)],
        'interest_skills': [('Occult', 30), ('Library Use', 25), ('Stealth', 25), ('Credit Rating', 20)],
    },
]


def _roll_3d6_x5() -> int:
    return sum(random.randint(1, 6) for _ in range(3)) * 5


def _roll_2d6_plus_6_x5() -> int:
    return (sum(random.randint(1, 6) for _ in range(2)) + 6) * 5


def _generate_coc_stats() -> dict[str, int]:
    stats = {k: _roll_3d6_x5() for k in COC_3D6_KEYS}
    stats.update({k: _roll_2d6_plus_6_x5() for k in COC_2D6_KEYS})
    return stats


def _score_archetype(stats: dict[str, int], weights: dict[str, float]) -> float:
    return sum(float(stats.get(k, 0)) * w for k, w in weights.items())


def _calc_derived(stats: dict[str, int]) -> dict[str, int]:
    hp = int((stats['CON'] + stats['SIZ']) / 5)
    mp = int(stats['POW'] / 5)
    san = int(stats['POW'])
    occ = int(stats['EDU'] * 4)
    interest = int(stats['INT'] * 2)
    return {
        'HP': hp,
        'MP': mp,
        'SAN': san,
        'occupation_skill_points': occ,
        'personal_interest_points': interest,
    }


def _alloc_points(total: int, weighted_skills: list[tuple[str, int]]) -> list[str]:
    weight_sum = sum(w for _, w in weighted_skills) or 1
    allocated = []
    used = 0
    for idx, (skill, weight) in enumerate(weighted_skills):
        if idx == len(weighted_skills) - 1:
            points = max(0, total - used)
        else:
            points = int(round(total * (weight / weight_sum)))
            used += points
        allocated.append(f'{skill}:{points}')
    return allocated


def _stats_to_line(stats: dict[str, int]) -> str:
    return ','.join([f'{k}:{int(stats[k])}' for k in COC_CORE_KEYS])


def _parse_stats_line(stats_line: str) -> dict[str, int] | None:
    parts = [p.strip() for p in stats_line.split(',') if p.strip()]
    parsed: dict[str, int] = {}
    for part in parts:
        if ':' not in part:
            return None
        key, raw = part.split(':', 1)
        key = key.strip().upper()
        raw = raw.strip()
        if key not in COC_CORE_KEYS:
            continue
        if not re.fullmatch(r'-?\d+', raw):
            return None
        parsed[key] = int(raw)
    if any(k not in parsed for k in COC_CORE_KEYS):
        return None
    return parsed


def _validate_coc_stats(stats: dict[str, int]) -> bool:
    for key in COC_3D6_KEYS:
        v = stats.get(key, 0)
        if v % 5 != 0 or v < 15 or v > 90:
            return False
    for key in COC_2D6_KEYS:
        v = stats.get(key, 0)
        if v % 5 != 0 or v < 40 or v > 90:
            return False
    return True


def _generate_coc_builds() -> list[dict[str, object]]:
    builds: list[dict[str, object]] = []
    used_lines: set[str] = set()
    for arch in COC_ARCHETYPES:
        best_stats = None
        best_score = float('-inf')
        for _ in range(200):
            candidate = _generate_coc_stats()
            score = _score_archetype(candidate, arch['weights'])  # type: ignore[arg-type]
            if score > best_score:
                best_score = score
                best_stats = candidate
        if best_stats is None:
            best_stats = _generate_coc_stats()
        line = _stats_to_line(best_stats)
        reroll_guard = 0
        while line in used_lines and reroll_guard < 50:
            best_stats = _generate_coc_stats()
            line = _stats_to_line(best_stats)
            reroll_guard += 1
        used_lines.add(line)
        derived = _calc_derived(best_stats)
        builds.append(
            {
                'archetype': arch['name'],
                'stats': best_stats,
                'line': line,
                'derived': derived,
                'occupation_suggested': _alloc_points(int(derived['occupation_skill_points']), arch['occupation_skills']),  # type: ignore[arg-type]
                'interest_suggested': _alloc_points(int(derived['personal_interest_points']), arch['interest_skills']),  # type: ignore[arg-type]
            }
        )
    return builds


def run_app() -> None:
    st.set_page_config(page_title='Script-Driven Narrative Agent', layout='wide')
    st.title('Script-Driven Narrative Agent System')

    if 'db' not in st.session_state:
        st.session_state.db = Database('narrative.db')
        st.session_state.vector = ChromaStore('.chroma')
        st.session_state.agent = NarrativeAgent(st.session_state.db, st.session_state.vector)
        st.session_state.messages = []
        st.session_state.last_retrieved = []
        st.session_state.shown_openings = set()

    db: Database = st.session_state.db
    vector: ChromaStore = st.session_state.vector
    agent: NarrativeAgent = st.session_state.agent
    if not hasattr(agent, 'set_debug_mode'):
        st.session_state.agent = NarrativeAgent(db, vector)
        agent = st.session_state.agent
    debug_mode = st.sidebar.toggle('Debug Prompt View', value=False)
    if hasattr(agent, 'set_debug_mode'):
        agent.set_debug_mode(debug_mode)
    else:
        setattr(agent, 'debug_mode', bool(debug_mode))

    state = db.get_system_state()
    existing_rule_items = [
        item for item in db.get_knowledge_by_type('rule') if item.get('metadata', {}).get('source') == 'database/GameRules.md'
    ]
    if not existing_rule_items:
        game_rules_knowledge = load_game_rules_knowledge()
        if game_rules_knowledge:
            db.insert_knowledge(game_rules_knowledge)
            vector.add_from_scenes([], knowledge=game_rules_knowledge)
            state = db.get_system_state()
    language_options = ['English', 'Chinese']
    current_language = state.get('output_language', 'English')
    if current_language not in language_options:
        current_language = 'English'
    selected_language = st.sidebar.selectbox(
        'Output Language',
        options=language_options,
        index=language_options.index(current_language),
    )
    if selected_language != state.get('output_language', 'English'):
        db.update_system_state({'output_language': selected_language})
        state = db.get_system_state()
    stage = state['stage']

    st.subheader(f'Stage: {stage}')

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown('### Current State')
        st.write(
            {
                'scene_id': state['current_scene_id'],
                'plot_id': state['current_plot_id'],
                'output_language': state.get('output_language', 'English'),
            }
        )
        st.progress(float(state['plot_progress']))
        st.caption('Plot Progress')
        st.progress(float(state['scene_progress']))
        st.caption('Scene Progress')
        if state.get('current_scene_intro'):
            st.info(state['current_scene_intro'])
        if debug_mode and st.session_state.last_retrieved:
            st.markdown('### Retrieved Knowledge')
            for doc in st.session_state.last_retrieved[:5]:
                st.write(f"- {doc.get('metadata', {}).get('type')}: {doc.get('content')}")

    with c1:
        if stage == 'upload':
            st.markdown('### 1) Upload Script')
            uploaded = st.file_uploader('Upload PDF script', type=['pdf'])
            start_page = st.number_input('Start Page', min_value=1, value=1)
            end_page = st.number_input('End Page (0 means auto)', min_value=0, value=0)
            story_start_page = st.number_input('Story Start Page (PDF page, 0 means auto)', min_value=0, value=0)
            story_end_page = st.number_input('Story End Page (PDF page, 0 means auto)', min_value=0, value=0)
            st.caption(
                'If Story Start/End are provided, that page span is parsed as story (scene/plot), and all other pages are parsed as knowledge.'
            )

            if uploaded and st.button('Parse Script'):
                try:
                    if (story_start_page > 0) != (story_end_page > 0):
                        st.error('Please fill both Story Start Page and Story End Page, or leave both as 0 for auto detection.')
                        st.stop()

                    read_start_page = int(start_page)
                    pages = read_pdf_pages(
                        uploaded.getvalue(),
                        start_page=read_start_page,
                        end_page=int(end_page) if end_page > 0 else None,
                    )

                    manual_story_start = None
                    manual_story_end = None
                    if story_start_page > 0 and story_end_page > 0:
                        manual_story_start = int(story_start_page) - read_start_page + 1
                        manual_story_end = int(story_end_page) - read_start_page + 1

                    bundle = parse_script_bundle(
                        pages,
                        story_start_page=manual_story_start,
                        story_end_page=manual_story_end,
                    )
                    scenes = bundle.get('scenes', [])
                    knowledge = bundle.get('knowledge', [])
                    game_rules_knowledge = load_game_rules_knowledge()
                    all_knowledge = knowledge + game_rules_knowledge
                    structure = bundle.get('structure', {})
                    parse_warnings = bundle.get('warnings', [])

                    if not scenes:
                        st.error('No scenes extracted. Please check the PDF content or LLM output.')
                        st.stop()

                    db.reset_story_data()
                    vector.reset()
                    db.insert_scenes(scenes)
                    db.insert_knowledge(all_knowledge)
                    vector.add_from_scenes(scenes, knowledge=all_knowledge)

                    db.save_summary('parse_structure', json.dumps(structure, ensure_ascii=False))
                    db.save_summary('parse_warnings', json.dumps(parse_warnings, ensure_ascii=False))
                    db.save_summary('parse_mode', bundle.get('parse_mode', 'balanced'))

                    first_scene = scenes[0]
                    first_plot = first_scene['plots'][0]
                    db.update_scene(first_scene['scene_id'], {'status': 'in_progress'})
                    db.update_system_state(
                        {
                            'stage': 'parse',
                            'current_scene_id': first_scene['scene_id'],
                            'current_plot_id': first_plot['plot_id'],
                            'plot_progress': 0.0,
                            'scene_progress': 0.0,
                            'current_scene_intro': '',
                        }
                    )

                    st.success(
                        f"Script parsed and stored. scenes={len(scenes)}, knowledge={len(all_knowledge)}, warnings={len(parse_warnings)}"
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f'Parse failed: {exc}')
                    st.stop()

        elif stage == 'parse':
            st.markdown('### 2) Review Scene Structure')

            scenes = db.list_scenes()
            plot_count = sum(len(scene.get('plots', [])) for scene in scenes)
            est_minutes = plot_count * 10

            if debug_mode:
                structure_raw = db.get_summary('parse_structure')
                if structure_raw:
                    try:
                        structure = json.loads(structure_raw)
                    except Exception:
                        structure = {'raw': structure_raw}
                    st.markdown('#### Parsed Document Structure')
                    st.write(structure)

                parse_mode = db.get_summary('parse_mode')
                if parse_mode:
                    st.caption(f'Parse mode: {parse_mode}')

                warnings_raw = db.get_summary('parse_warnings')
                if warnings_raw:
                    try:
                        parse_warnings = json.loads(warnings_raw)
                    except Exception:
                        parse_warnings = [warnings_raw]
                    if parse_warnings:
                        st.warning(f'Parser warnings: {len(parse_warnings)}')
                        for w in parse_warnings[:8]:
                            st.write(f'- {w}')

                for scene in scenes:
                    st.write(
                        {
                            'scene_id': scene['scene_id'],
                            'scene_goal': scene['scene_goal'],
                            'source_pages': f"{scene.get('source_page_start', '?')}-{scene.get('source_page_end', '?')}",
                            'scene_description': (scene.get('scene_description', '') or '')[:120],
                            'plots': len(scene.get('plots', [])),
                        }
                    )

                knowledge_items = db.list_knowledge()
                if knowledge_items:
                    counts: dict[str, int] = {}
                    for item in knowledge_items:
                        t = item.get('knowledge_type', 'other')
                        counts[t] = counts.get(t, 0) + 1
                    st.markdown('#### Knowledge Overview')
                    st.write(counts)
            else:
                st.write(
                    {
                        'scene_count': len(scenes),
                        'plot_count': plot_count,
                        'estimated_play_time_minutes': est_minutes,
                    }
                )
                st.caption(f'estimated play time {est_minutes} minutes')

            if st.button('Confirm Structure'):
                db.update_system_state({'stage': 'character'})
                st.rerun()

        elif stage == 'character':
            st.markdown('### 3) Create Character')

            if 'coc_builds' not in st.session_state:
                st.session_state.coc_builds = _generate_coc_builds()
            if 'character_stats_line' not in st.session_state:
                st.session_state.character_stats_line = st.session_state.coc_builds[0]['line']
            if 'selected_archetype_name' not in st.session_state:
                st.session_state.selected_archetype_name = st.session_state.coc_builds[0]['archetype']
            if 'occupation_alloc_text' not in st.session_state:
                st.session_state.occupation_alloc_text = ''
            if 'interest_alloc_text' not in st.session_state:
                st.session_state.interest_alloc_text = ''

            archetype_to_build = {str(b['archetype']): b for b in st.session_state.coc_builds}
            archetype_names = list(archetype_to_build.keys())
            selected_archetype_name = st.session_state.selected_archetype_name
            if selected_archetype_name not in archetype_to_build:
                selected_archetype_name = archetype_names[0]
                st.session_state.selected_archetype_name = selected_archetype_name
            selected_build = archetype_to_build[selected_archetype_name]

            st.markdown('#### Step 1: Character Identity')
            st.caption('You must fill in both fields to continue.')
            name = st.text_input('Name', key='character_name_input')
            background = st.text_area('Background', key='character_background_input')

            st.markdown('#### Step 2: Choose Characteristic Build')
            st.caption(
                'Pick one of the 10 generated builds, then copy-paste into the Characteristics box. '
                'You can edit numbers afterward.'
            )
            st.caption('Rules: STR/CON/DEX/APP/POW = 15-90 (step 5), SIZ/INT/EDU = 40-90 (step 5).')

            build_options = [f"{b['archetype']} | {b['line']}" for b in st.session_state.coc_builds]
            chosen_build_label = st.selectbox('Generated Builds (10)', options=build_options, key='build_pick_label')
            chosen_build_line = chosen_build_label.split(' | ', 1)[1]
            if st.button('Apply Build to Characteristics'):
                st.session_state.character_stats_line = chosen_build_line

            st.code('\n'.join([b['line'] for b in st.session_state.coc_builds]), language='text')

            stats = st.text_input('Characteristics', key='character_stats_line')
            parsed_stats = _parse_stats_line(stats)
            stats_valid = bool(parsed_stats and _validate_coc_stats(parsed_stats))
            if stats_valid:
                derived = _calc_derived(parsed_stats)
                st.write(
                    {
                        'HP': derived['HP'],
                        'MP': derived['MP'],
                        'SAN': derived['SAN'],
                        'occupation_skill_points': derived['occupation_skill_points'],
                        'personal_interest_points': derived['personal_interest_points'],
                    }
                )
            else:
                st.warning(
                    'Invalid characteristics. Use format: STR:65,CON:50,SIZ:60,DEX:80,APP:60,INT:75,POW:75,EDU:70'
                )

            st.markdown('#### Step 3: Select Archetype')
            selected_archetype_name = st.selectbox('Archetype', options=archetype_names, key='selected_archetype_name')
            selected_build = archetype_to_build[selected_archetype_name]
            st.caption(f"Suggested profile focus: {selected_archetype_name}")

            st.markdown('#### Step 4: Allocate Skills')
            st.caption('Use the suggested allocation below as a starting point, then edit freely.')
            occ_default = '\n'.join(selected_build['occupation_suggested'])
            interest_default = '\n'.join(selected_build['interest_suggested'])
            if not st.session_state.occupation_alloc_text:
                st.session_state.occupation_alloc_text = occ_default
            if not st.session_state.interest_alloc_text:
                st.session_state.interest_alloc_text = interest_default
            if st.button('Use Suggested Skills for Selected Archetype'):
                st.session_state.occupation_alloc_text = occ_default
                st.session_state.interest_alloc_text = interest_default

            occupation_alloc = st.text_area(
                'Occupation Skills (one per line, e.g., Spot Hidden:60)',
                key='occupation_alloc_text',
                height=160,
            )
            interest_alloc = st.text_area(
                'Personal Interest Skills (one per line, e.g., Occult:40)',
                key='interest_alloc_text',
                height=130,
            )

            if st.button('Save Character'):
                if not name.strip():
                    st.error('Name is required.')
                    st.stop()
                if not background.strip():
                    st.error('Background is required.')
                    st.stop()

                parsed = _parse_stats_line(stats)
                if not parsed or not _validate_coc_stats(parsed):
                    st.error('Please provide a valid CoC characteristic line before saving.')
                    st.stop()

                derived = _calc_derived(parsed)
                profile = {
                    'name': name.strip(),
                    'background': background.strip(),
                    'characteristics': parsed,
                    'derived_attributes': {
                        'HP': derived['HP'],
                        'MP': derived['MP'],
                        'SAN': derived['SAN'],
                    },
                    'skill_points': {
                        'occupation': derived['occupation_skill_points'],
                        'personal_interest': derived['personal_interest_points'],
                    },
                    'selected_archetype': selected_archetype_name,
                    'suggested_skill_allocations': {
                        'occupation': selected_build['occupation_suggested'],
                        'personal_interest': selected_build['interest_suggested'],
                    },
                    'chosen_skill_allocations': {
                        'occupation': [line.strip() for line in occupation_alloc.splitlines() if line.strip()],
                        'personal_interest': [line.strip() for line in interest_alloc.splitlines() if line.strip()],
                    },
                }
                db.save_player_profile(profile)
                db.update_system_state({'stage': 'session'})
                st.rerun()

        else:
            st.markdown('### 4) Narrative Session')
            opening_key = state['current_scene_id']
            opening_text = agent.ensure_kp_opening(state['current_scene_id'], state['current_plot_id'])
            if opening_text and opening_key not in st.session_state.shown_openings:
                st.session_state.messages.append(
                    {
                        'user': '',
                        'agent': opening_text,
                        'dice': None,
                        'debug_prompts': list(getattr(agent, 'latest_debug_prompts', [])),
                    }
                )
                st.session_state.shown_openings.add(opening_key)

            for turn in st.session_state.messages:
                if turn['user']:
                    st.chat_message('user').write(turn['user'])
                st.chat_message('assistant').write(turn['agent'])
                if turn.get('dice'):
                    st.caption(f"Dice: {turn['dice']}")
                if debug_mode and turn.get('debug_prompts'):
                    with st.expander('Debug Prompts', expanded=False):
                        for idx, item in enumerate(turn.get('debug_prompts', []), start=1):
                            st.caption(f"{idx}. {item.get('name', 'prompt')}")
                            st.code(item.get('prompt', ''), language='text')

            user_msg = st.chat_input('Describe your action...')
            if user_msg:
                result = agent.run_turn(user_msg)
                st.session_state.last_retrieved = result.get('retrieved_docs', [])
                st.session_state.messages.append(
                    {
                        'user': user_msg,
                        'agent': result.get('response', ''),
                        'dice': result.get('dice_result'),
                        'debug_prompts': result.get('debug_prompts', []),
                    }
                )
                st.rerun()
