from __future__ import annotations

import json
import random
import re

import streamlit as st

from app.agent_graph import NarrativeAgent
from app.database import Database
from app.parser import detect_source_type, parse_script_bundle, read_uploaded_document
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


def _inject_demo_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --um-blue: #00274c;
            --um-blue-soft: #5a6d82;
            --um-maize: #d7b14a;
            --um-ink: #1f2d3a;
            --um-ink-strong: #14212e;
            --um-panel: rgba(255, 253, 248, 0.94);
            --um-panel-soft: rgba(252, 249, 242, 0.92);
            --um-code-bg: #161b26;
            --um-code-ink: #f3f7fb;
            --um-paper: #f5f1e8;
            --um-line: rgba(0, 39, 76, 0.12);
        }

        html, body, [class*="css"] {
            font-family: Georgia, "Times New Roman", serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f7f4ec 0%, #f2efe7 100%);
            color: var(--um-ink);
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div,
        .stMarkdown,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--um-ink-strong);
        }

        [data-testid="stHeader"] {
            background: rgba(245, 241, 232, 0.9);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(245, 241, 232, 0.98), rgba(239, 234, 222, 0.98));
            color: var(--um-ink-strong);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] button {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stToggle label,
        [data-testid="stSidebar"] [role="switch"] + div,
        [data-testid="stSidebar"] [role="switch"] ~ * {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stSidebar"] [role="switch"] {
            background: #efe4c6 !important;
            border: 1px solid rgba(0, 39, 76, 0.28) !important;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.45);
        }

        [data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
            background: #d7b14a !important;
            border-color: rgba(0, 39, 76, 0.45) !important;
        }

        [data-testid="stSidebar"] [role="switch"] > div,
        [data-testid="stSidebar"] [role="switch"] [data-testid="stThumbValue"] {
            background: #16304d !important;
            color: #16304d !important;
        }

        .block-container {
            max-width: 880px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .gm-section {
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }

        .gm-section-eyebrow {
            color: rgba(31, 45, 58, 0.82);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .gm-section-title {
            font-size: 1.18rem;
            color: var(--um-blue);
            margin: 0 0 0.12rem 0;
            font-weight: 700;
        }

        .gm-section-copy {
            color: var(--um-ink-strong);
            font-size: 0.93rem;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stMetric"] {
            background: var(--um-panel);
            border: 1px solid var(--um-line);
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            box-shadow: none;
        }

        div[data-testid="stMetric"] label {
            color: rgba(31, 45, 58, 0.82) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--um-blue) !important;
            font-size: 1.1rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(0, 39, 76, 0.18);
            background: rgba(215, 177, 74, 0.32);
            color: var(--um-ink-strong);
            font-weight: 700;
            padding: 0.45rem 0.85rem;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(0, 39, 76, 0.28);
            background: rgba(215, 177, 74, 0.42);
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stFileUploader section,
        [data-testid="stChatInput"] {
            border-radius: 8px !important;
            border-color: rgba(0, 39, 76, 0.15) !important;
            background: var(--um-panel) !important;
            color: var(--um-ink-strong) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #5b6673 !important;
        }

        .stTextInput label,
        .stTextArea label,
        .stNumberInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stRadio label,
        .stCheckbox label,
        .stCaption,
        [data-testid="stWidgetLabel"],
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--um-ink-strong) !important;
        }

        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] *,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            color: var(--um-ink-strong) !important;
        }

        [data-baseweb="menu"] *,
        [role="listbox"] *,
        [role="option"] {
            color: var(--um-ink-strong) !important;
            background: #fffdf8 !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {
            border-color: rgba(215, 177, 74, 0.85) !important;
            box-shadow: 0 0 0 1px rgba(215, 177, 74, 0.45) !important;
        }

        [data-testid="stExpander"] {
            border-radius: 8px;
            border: 1px solid var(--um-line);
            background: var(--um-panel-soft);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] details > div,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
        [data-testid="stExpander"] summary *,
        [data-testid="stAlert"] *,
        [data-testid="stChatMessage"] * {
            color: var(--um-ink-strong) !important;
        }

        [data-testid="stExpander"] summary {
            background: rgba(247, 241, 229, 0.92) !important;
        }

        [data-testid="stExpander"] details > div {
            background: rgba(255, 251, 244, 0.96) !important;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--um-line);
            background: rgba(255, 251, 244, 0.96);
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 252, 247, 0.62);
            border: none;
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.6rem;
            box-shadow: none;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.8;
            font-size: 1rem;
        }

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, rgba(215, 177, 74, 0.8), rgba(0, 39, 76, 0.45)) !important;
        }

        .stCodeBlock,
        .stCode,
        [data-testid="stCode"],
        [data-testid="stCodeBlock"] {
            background: transparent !important;
        }

        .stCodeBlock pre,
        .stCode pre,
        [data-testid="stCode"] pre,
        [data-testid="stCodeBlock"] pre {
            background: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 0.95rem 1rem !important;
            margin: 0.35rem 0 0 0 !important;
            overflow-x: auto !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            box-shadow: none !important;
        }

        .stCodeBlock pre code,
        .stCode pre code,
        [data-testid="stCode"] pre code,
        [data-testid="stCodeBlock"] pre code {
            display: block !important;
            background: transparent !important;
            color: #f8fafc !important;
            font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace !important;
            font-size: 0.92rem !important;
            line-height: 1.58 !important;
            text-shadow: none !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        .stCodeBlock pre code *,
        .stCode pre code *,
        [data-testid="stCode"] pre code *,
        [data-testid="stCodeBlock"] pre code * {
            background: transparent !important;
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
            text-shadow: none !important;
            opacity: 1 !important;
            border: none !important;
            box-shadow: none !important;
        }

        .stCaption {
            color: #344557;
        }

        button[kind="secondary"],
        button[kind="secondary"] *,
        [data-baseweb="tab-list"] *,
        [data-baseweb="tab"] *,
        [role="tab"] * {
            color: var(--um-ink-strong) !important;
        }

        [data-baseweb="tab"] {
            background: rgba(248, 243, 233, 0.94) !important;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: rgba(215, 177, 74, 0.26) !important;
        }

        .gm-statusline {
            position: sticky;
            top: 0.35rem;
            z-index: 10;
            margin: 0.2rem 0 1rem 0;
            padding: 0.4rem 0;
            color: var(--um-ink-strong);
            font-size: 0.92rem;
            background: linear-gradient(180deg, rgba(247, 244, 236, 0.95), rgba(247, 244, 236, 0.82));
            backdrop-filter: blur(4px);
        }

        .gm-settings {
            margin: 0 0 0.9rem 0;
        }

        .gm-loading-shell {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            gap: 0.55rem;
            color: var(--um-ink-strong);
        }

        .gm-parse-overlay {
            position: fixed;
            inset: 0;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, rgba(247, 244, 236, 0.985), rgba(242, 239, 231, 0.985));
            text-align: center;
        }

        .gm-loading-inline {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--um-ink-strong);
            padding: 0.2rem 0;
        }

        .gm-loading-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.25rem;
            height: 1.25rem;
            color: rgba(215, 177, 74, 0.9);
            font-size: 1rem;
            animation: gm-glow 1.8s ease-in-out infinite;
        }

        .gm-loading-text {
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .gm-loading-dots {
            display: inline-flex;
            align-items: center;
            gap: 0.18rem;
        }

        .gm-loading-dots span {
            width: 0.22rem;
            height: 0.22rem;
            border-radius: 50%;
            background: rgba(0, 39, 76, 0.42);
            animation: gm-pulse 1.4s ease-in-out infinite;
        }

        .gm-loading-dots span:nth-child(2) {
            animation-delay: 0.18s;
        }

        .gm-loading-dots span:nth-child(3) {
            animation-delay: 0.36s;
        }

        @keyframes gm-pulse {
            0%, 80%, 100% {
                opacity: 0.25;
                transform: translateY(0);
            }
            40% {
                opacity: 0.85;
                transform: translateY(-1px);
            }
        }

        @keyframes gm-glow {
            0%, 100% {
                opacity: 0.45;
                transform: scale(0.96);
            }
            50% {
                opacity: 0.95;
                transform: scale(1.05);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_header(title: str, eyebrow: str, copy: str = '') -> None:
    copy_html = f'<div class="gm-section-copy">{copy}</div>' if copy else ''
    st.markdown(
        f"""
        <div class="gm-section">
            <div class="gm-section-eyebrow">{eyebrow}</div>
            <div class="gm-section-title">{title}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_line(state: dict[str, object]) -> None:
    bits = []
    if state.get('current_scene_id'):
        bits.append(f"Scene {state['current_scene_id']}")
    if state.get('current_plot_id'):
        bits.append(f"Plot {state['current_plot_id']}")
    if state.get('output_language'):
        bits.append(str(state['output_language']))
    if bits:
        st.markdown(f"<div class='gm-statusline'>{' / '.join(bits)}</div>", unsafe_allow_html=True)


def _render_settings(debug_mode: bool, current_language: str) -> None:
    st.markdown("<div class='gm-settings'></div>", unsafe_allow_html=True)
    with st.expander('Settings', expanded=False):
        st.selectbox('Output Language', options=['English', 'Chinese'], index=['English', 'Chinese'].index(current_language), key='output_language_select')


def _render_loading_state(target: object, text: str, centered: bool = False) -> None:
    if centered:
        target.markdown(
            f"""
            <div class="gm-parse-overlay">
                <div class="gm-loading-shell">
                    <div class="gm-loading-icon">✦</div>
                    <div class="gm-loading-text">{text}</div>
                    <div class="gm-loading-dots" aria-hidden="true">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    target.markdown(f"*{text}*  \n`...`")


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


def _load_messages_from_db(db: Database) -> list[dict[str, object]]:
    rows = db.conn.execute(
        'SELECT scene_id, plot_id, user, agent FROM memory ORDER BY id ASC'
    ).fetchall()
    messages: list[dict[str, object]] = []
    for row in rows:
        user = row['user'] or ''
        agent = row['agent'] or ''
        messages.append(
            {
                'user': user,
                'agent': agent,
                'dice': None,
                'skill_check': None,
                'debug_prompts': [],
            }
        )
    return messages


def run_app() -> None:
    st.set_page_config(page_title='Script-Driven Narrative Agent', layout='wide')
    _inject_demo_theme()

    if 'db' not in st.session_state:
        st.session_state.db = Database('narrative.db')
        st.session_state.vector = ChromaStore('.chroma')
        st.session_state.agent = NarrativeAgent(st.session_state.db, st.session_state.vector)
        st.session_state.messages = _load_messages_from_db(st.session_state.db)
        st.session_state.last_retrieved = []

    db: Database = st.session_state.db
    vector: ChromaStore = st.session_state.vector
    agent: NarrativeAgent = st.session_state.agent
    if not hasattr(agent, 'set_debug_mode'):
        st.session_state.agent = NarrativeAgent(db, vector)
        agent = st.session_state.agent

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
    debug_mode = st.sidebar.toggle('Debug Prompt View', value=bool(st.session_state.get('debug_prompt_toggle', False)), key='debug_prompt_toggle')
    _render_settings(debug_mode, current_language)
    debug_mode = bool(st.session_state.get('debug_prompt_toggle', debug_mode))
    selected_language = st.session_state.get('output_language_select', current_language)
    if hasattr(agent, 'set_debug_mode'):
        agent.set_debug_mode(debug_mode)
    else:
        setattr(agent, 'debug_mode', bool(debug_mode))
    if selected_language != state.get('output_language', 'English'):
        db.update_system_state({'output_language': selected_language})
        state = db.get_system_state()
    stage = state['stage']
    if stage == 'session' and not st.session_state.messages:
        restored_messages = _load_messages_from_db(db)
        st.session_state.messages = restored_messages
        if not restored_messages and state.get('current_scene_id') and state.get('current_plot_id'):
            initial_result = agent.generate_initial_response()
            st.session_state.last_retrieved = initial_result.get('retrieved_docs', [])
            st.session_state.messages.append(
                {
                    'user': '',
                    'agent': initial_result.get('response', ''),
                    'dice': initial_result.get('dice_result'),
                    'skill_check': initial_result.get('skill_check_result'),
                    'debug_prompts': initial_result.get('debug_prompts', []),
                }
            )
            st.rerun()

    if stage != 'session':
        _render_status_line(state)

    if stage == 'upload':
        upload_panel = st.container()
        uploaded = None
        source_type = 'pdf'
        source_unit_label = 'page'
        with upload_panel:
            _render_section_header('Upload Script', 'Step 1')
            uploaded = st.file_uploader('Upload script (.pdf, .md, .markdown)', type=['pdf', 'md', 'markdown'])
            unsupported_type = False
            if uploaded:
                try:
                    source_type = detect_source_type(uploaded.name, getattr(uploaded, 'type', None))
                except ValueError:
                    unsupported_type = True
                    st.error('Unsupported file type. Please upload a PDF or Markdown file.')
                source_unit_label = 'line' if source_type == 'markdown' else 'page'
                st.caption(f"Detected source type: {source_type}. Range controls below use {source_unit_label}s.")
            unit_title = source_unit_label.title()
            start_page = st.number_input(f'Start {unit_title}', min_value=1, value=1)
            end_page = st.number_input(f'End {unit_title} (0 means auto)', min_value=0, value=0)
            story_start_page = st.number_input(
                f'Story Start {unit_title} ({source_unit_label} number, 0 means auto)',
                min_value=0,
                value=0,
            )
            story_end_page = st.number_input(
                f'Story End {unit_title} ({source_unit_label} number, 0 means auto)',
                min_value=0,
                value=0,
            )
            st.caption(f'Set the story {source_unit_label} range only if you want to override auto detection.')
            parse_clicked = bool(uploaded and st.button('Parse Script'))

        if parse_clicked:
            upload_panel.empty()
            parse_loading = st.empty()
            try:
                _render_loading_state(parse_loading, 'Parsing the script...', centered=True)
                if (story_start_page > 0) != (story_end_page > 0):
                    parse_loading.empty()
                    st.error(
                        f'Please fill both Story Start {unit_title} and Story End {unit_title}, or leave both as 0 for auto detection.'
                    )
                    st.stop()

                if unsupported_type:
                    parse_loading.empty()
                    st.error('Unsupported file type. Please upload a PDF or Markdown file.')
                    st.stop()

                start_unit = int(start_page)
                end_unit = int(end_page) if end_page > 0 else None
                document = read_uploaded_document(
                    uploaded.name,
                    uploaded.getvalue(),
                    start_unit=start_unit,
                    end_unit=end_unit,
                    mime_type=getattr(uploaded, 'type', None),
                )
                if not document.segments:
                    parse_loading.empty()
                    st.error(f'No readable {source_unit_label}s found in the uploaded file.')
                    st.stop()

                manual_story_start = None
                manual_story_end = None
                if story_start_page > 0 and story_end_page > 0:
                    manual_story_start = int(story_start_page)
                    manual_story_end = int(story_end_page)
                    if (
                        manual_story_start < document.display_start
                        or manual_story_end > document.display_end
                        or manual_story_start > document.display_end
                        or manual_story_end < document.display_start
                    ):
                        parse_loading.empty()
                        st.error(
                            f'Story {source_unit_label} range must stay within the selected {source_unit_label} range '
                            f'({document.display_start}-{document.display_end}).'
                        )
                        st.stop()

                _render_loading_state(parse_loading, 'Organizing scenes...', centered=True)
                bundle = parse_script_bundle(
                    source_document=document,
                    story_start_page=manual_story_start,
                    story_end_page=manual_story_end,
                )
                scenes = bundle.get('scenes', [])
                knowledge = bundle.get('knowledge', [])
                game_rules_knowledge = load_game_rules_knowledge()
                all_knowledge = knowledge + game_rules_knowledge
                structure = bundle.get('structure', {})
                parse_warnings = bundle.get('warnings', [])
                source_metadata = bundle.get('source_metadata', {})

                if not scenes:
                    parse_loading.empty()
                    st.error(f'No scenes extracted. Please check the {source_type} content or LLM output.')
                    st.stop()

                _render_loading_state(parse_loading, 'Preparing the world...', centered=True)
                db.reset_story_data()
                vector.reset()
                db.insert_scenes(scenes)
                db.insert_knowledge(all_knowledge)
                vector.add_from_scenes(scenes, knowledge=all_knowledge)

                db.save_summary('parse_structure', json.dumps(structure, ensure_ascii=False))
                db.save_summary('parse_warnings', json.dumps(parse_warnings, ensure_ascii=False))
                db.save_summary('parse_mode', bundle.get('parse_mode', 'balanced'))
                db.save_summary('parse_source_meta', json.dumps(source_metadata, ensure_ascii=False))

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

                parse_loading.empty()
                st.success(
                    f"Script parsed and stored. scenes={len(scenes)}, knowledge={len(all_knowledge)}, warnings={len(parse_warnings)}"
                )
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                parse_loading.empty()
                st.error(f'Parse failed: {exc}')
                st.stop()

    elif stage == 'parse':
        _render_section_header('Review Structure', 'Step 2')

        scenes = db.list_scenes()
        plot_count = sum(len(scene.get('plots', [])) for scene in scenes)
        est_minutes = plot_count * 10
        source_meta_raw = db.get_summary('parse_source_meta')
        try:
            source_meta = json.loads(source_meta_raw) if source_meta_raw else {}
        except Exception:
            source_meta = {}
        source_unit_label = str(source_meta.get('source_unit_label', 'page') or 'page')
        source_unit_title = source_unit_label.title()

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

            if source_meta:
                st.markdown('#### Source Metadata')
                st.write(source_meta)

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
                        f'source_{source_unit_label}s': (
                            f"{source_unit_title} "
                            f"{scene.get('source_page_start', '?')}-{scene.get('source_page_end', '?')}"
                        ),
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
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric('Scenes', len(scenes))
            with m2:
                st.metric('Plots', plot_count)
            with m3:
                st.metric('Est. Minutes', est_minutes)
            st.caption(f'estimated play time {est_minutes} minutes')

        if st.button('Confirm Structure'):
            db.update_system_state({'stage': 'character'})
            st.rerun()

    elif stage == 'character':
        _render_section_header('Character', 'Step 3')

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

        _render_section_header('Step 1: Character Identity', 'Required')
        id_col1, id_col2 = st.columns([1, 1.2])
        with id_col1:
            name = st.text_input('Name', key='character_name_input')
        with id_col2:
            background = st.text_area('Background', key='character_background_input', height=120)

        _render_section_header('Step 2: Characteristic Build', 'Step 2')
        st.caption('STR/CON/DEX/APP/POW: 15-90. SIZ/INT/EDU: 40-90.')

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
            d1, d2, d3, d4, d5 = st.columns(5)
            with d1:
                st.metric('HP', derived['HP'])
            with d2:
                st.metric('MP', derived['MP'])
            with d3:
                st.metric('SAN', derived['SAN'])
            with d4:
                st.metric('Occupation', derived['occupation_skill_points'])
            with d5:
                st.metric('Interest', derived['personal_interest_points'])
        else:
            st.warning(
                'Invalid characteristics. Use format: STR:65,CON:50,SIZ:60,DEX:80,APP:60,INT:75,POW:75,EDU:70'
            )

        _render_section_header('Step 3: Archetype', 'Step 3')
        selected_archetype_name = st.selectbox('Archetype', options=archetype_names, key='selected_archetype_name')
        selected_build = archetype_to_build[selected_archetype_name]

        _render_section_header('Step 4: Skills', 'Step 4')
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
        for turn in st.session_state.messages:
            if turn['user']:
                st.chat_message('user').write(turn['user'])
            st.chat_message('assistant').write(turn['agent'])
            if turn.get('dice'):
                st.caption(f"Dice Roll Result: {turn['dice']}")
            if turn.get('skill_check'):
                st.caption(f"Skill Check Result: {turn['skill_check']}")
            if debug_mode and turn.get('debug_prompts'):
                with st.expander('Debug Prompts', expanded=False):
                    for idx, item in enumerate(turn.get('debug_prompts', []), start=1):
                        label = f"{idx}. {item.get('name', 'prompt')}"
                        with st.expander(label, expanded=False):
                            st.code(item.get('prompt', ''), language='text')

        user_msg = st.chat_input('Describe your action...')
        if user_msg:
            st.chat_message('user').write(user_msg)
            thinking_placeholder = st.chat_message('assistant').empty()
            thinking_texts = [
                'The Keeper is thinking...',
                'Weaving the next scene...',
                'Something is unfolding...',
            ]
            _render_loading_state(
                thinking_placeholder,
                thinking_texts[len(st.session_state.messages) % len(thinking_texts)],
            )
            result = agent.run_turn(user_msg)
            thinking_placeholder.empty()
            st.session_state.last_retrieved = result.get('retrieved_docs', [])
            st.session_state.messages.append(
                {
                    'user': user_msg,
                    'agent': result.get('response', ''),
                    'dice': result.get('dice_result'),
                    'skill_check': result.get('skill_check_result'),
                    'debug_prompts': result.get('debug_prompts', []),
                }
            )
            st.rerun()
        if debug_mode and st.session_state.last_retrieved:
            with st.expander('Retrieved Knowledge', expanded=False):
                for doc in st.session_state.last_retrieved[:5]:
                    st.write(f"- {doc.get('metadata', {}).get('type')}: {doc.get('content')}")



