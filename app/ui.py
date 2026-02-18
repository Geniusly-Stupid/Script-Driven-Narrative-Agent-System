from __future__ import annotations

import streamlit as st

from app.agent_graph import NarrativeAgent
from app.database import Database
from app.parser import parse_script, read_pdf_pages
from app.vector_store import ChromaStore


def run_app() -> None:
    st.set_page_config(page_title='Script-Driven Narrative Agent', layout='wide')
    st.title('Script-Driven Narrative Agent System')

    if 'db' not in st.session_state:
        st.session_state.db = Database('narrative.db')
        st.session_state.vector = ChromaStore('.chroma')
        st.session_state.agent = NarrativeAgent(st.session_state.db, st.session_state.vector)
        st.session_state.messages = []
        st.session_state.last_retrieved = []

    db: Database = st.session_state.db
    vector: ChromaStore = st.session_state.vector
    agent: NarrativeAgent = st.session_state.agent

    state = db.get_system_state()
    stage = state['stage']

    st.subheader(f'Stage: {stage}')

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown('### Current State')
        st.write({'scene_id': state['current_scene_id'], 'plot_id': state['current_plot_id']})
        st.progress(float(state['plot_progress']))
        st.caption('Plot Progress')
        st.progress(float(state['scene_progress']))
        st.caption('Scene Progress')
        if state.get('current_scene_intro'):
            st.info(state['current_scene_intro'])
        if st.session_state.last_retrieved:
            st.markdown('### Retrieved Knowledge')
            for doc in st.session_state.last_retrieved[:5]:
                st.write(f"- {doc.get('metadata', {}).get('type')}: {doc.get('content')}")

    with c1:
        if stage == 'upload':
            st.markdown('### 1) Upload Script')
            uploaded = st.file_uploader('Upload PDF script', type=['pdf'])
            start_page = st.number_input('Start Page', min_value=1, value=1)
            end_page = st.number_input('End Page (0 means auto)', min_value=0, value=0)
            if uploaded and st.button('Parse Script'):
                pages = read_pdf_pages(uploaded.getvalue(), start_page=int(start_page), end_page=int(end_page) if end_page > 0 else None)
                scenes = parse_script(pages)
                db.reset_story_data()
                vector.reset()
                db.insert_scenes(scenes)
                vector.add_from_scenes(scenes)
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
                st.success('Script parsed and stored.')
                st.rerun()

        elif stage == 'parse':
            st.markdown('### 2) Review Scene Structure')
            for scene in db.list_scenes():
                st.write({'scene_id': scene['scene_id'], 'scene_goal': scene['scene_goal'], 'plots': len(scene.get('plots', []))})
            if st.button('Confirm Structure'):
                db.update_system_state({'stage': 'character'})
                st.rerun()

        elif stage == 'character':
            st.markdown('### 3) Create Character')
            name = st.text_input('Name')
            background = st.text_area('Background')
            traits = st.text_input('Traits (comma-separated)')
            stats = st.text_input('Stats (e.g. str:8,int:10)')
            skills = st.text_input('Special Skills (comma-separated)')
            if st.button('Save Character'):
                stat_map = {}
                for part in [p.strip() for p in stats.split(',') if p.strip() and ':' in p]:
                    k, v = part.split(':', 1)
                    try:
                        stat_map[k.strip()] = float(v.strip())
                    except ValueError:
                        stat_map[k.strip()] = 0
                profile = {
                    'name': name,
                    'background': background,
                    'traits': [t.strip() for t in traits.split(',') if t.strip()],
                    'stats': stat_map,
                    'special_skills': [s.strip() for s in skills.split(',') if s.strip()],
                }
                db.save_player_profile(profile)
                db.update_system_state({'stage': 'session'})
                st.rerun()

        else:
            st.markdown('### 4) Narrative Session')
            for turn in st.session_state.messages:
                st.chat_message('user').write(turn['user'])
                st.chat_message('assistant').write(turn['agent'])
                if turn.get('dice'):
                    st.caption(f"Dice: {turn['dice']}")

            user_msg = st.chat_input('Describe your action...')
            if user_msg:
                result = agent.run_turn(user_msg)
                st.session_state.last_retrieved = result.get('retrieved_docs', [])
                st.session_state.messages.append(
                    {'user': user_msg, 'agent': result.get('response', ''), 'dice': result.get('dice_result')}
                )
                st.rerun()
