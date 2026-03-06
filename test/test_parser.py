import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser import parse_script, parse_script_bundle


def _pages_from_prompt(prompt: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"\[PAGE\s+(\d+)\]", prompt)]


def _mock_llm(prompt: str) -> str:
    if 'TASK: IDENTIFY_SCRIPT_STRUCTURE_LABELS' in prompt:
        labels = []
        for page_no in _pages_from_prompt(prompt):
            if page_no <= 2:
                label = 'front'
            elif page_no <= 6:
                label = 'story'
            else:
                label = 'appendix'
            labels.append({'page': page_no, 'label': label, 'confidence': 0.92})
        return json.dumps({'page_labels': labels}, ensure_ascii=False)

    if 'TASK: IDENTIFY_SCRIPT_STRUCTURE' in prompt:
        return json.dumps({'story': {'start_page': 3, 'end_page': 6}}, ensure_ascii=False)

    if 'TASK: EXTRACT_KNOWLEDGE_ONLY' in prompt:
        if 'Phase: front_knowledge' in prompt:
            return json.dumps(
                {
                    'knowledge': [
                        {
                            'knowledge_type': 'setting',
                            'title': 'Module Setup',
                            'content': 'Town setup and campaign constraints.',
                            'source_page_start': 1,
                            'source_page_end': 1,
                        },
                        {
                            'knowledge_type': 'npc',
                            'title': 'Important NPC',
                            'content': 'Key motivations for preface NPC.',
                            'source_page_start': 2,
                            'source_page_end': 2,
                        },
                    ]
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                'knowledge': [
                    {
                        'knowledge_type': 'rule',
                        'title': 'Appendix Rule',
                        'content': 'Optional sanity variant at the end.',
                        'source_page_start': 7,
                        'source_page_end': 8,
                    }
                ]
            },
            ensure_ascii=False,
        )

    if 'TASK: EXTRACT_STORY_SCENES' in prompt:
        return json.dumps(
            {
                'scenes': [
                    {
                        'scene_goal': 'Investigate first incident',
                        'scene_description': 'The party reaches the square and interviews witnesses. Conflicting accounts push them toward the river and reveal faction pressure. They need to decide whether to trust local militia or follow unofficial clues.',
                        'scene_summary': 'Opening investigation beat.',
                        'scene_type': 'normal',
                        'source_page_start': 3,
                        'source_page_end': 5,
                        'plots': [
                            {
                                'plot_goal': 'Collect witness account',
                                'mandatory_events': ['Interview witness'],
                                'npc': ['Militia Liaison'],
                                'locations': ['Town Square'],
                                'source_page_start': 3,
                                'source_page_end': 5,
                            }
                        ],
                    },
                    {
                        'scene_goal': 'Move to next location',
                        'scene_description': 'The team departs quickly before nightfall.',
                        'scene_summary': 'Bridge transition.',
                        'scene_type': 'transition',
                        'source_page_start': 6,
                        'source_page_end': 6,
                        'plots': [
                            {
                                'plot_goal': 'Travel to river road',
                                'mandatory_events': ['Leave square'],
                                'npc': [],
                                'locations': ['Road'],
                                'source_page_start': 6,
                                'source_page_end': 6,
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        )

    if 'TASK: REFINE_SCENE_PLOTS' in prompt:
        return json.dumps(
            {
                'scene_type': 'normal',
                'plots': [
                    {
                        'plot_goal': 'Interview locals and collect baseline testimony',
                        'mandatory_events': ['Interview witness', 'Cross-check statements'],
                        'npc': ['Militia Liaison', 'Vendor'],
                        'locations': ['Town Square'],
                        'source_page_start': 3,
                        'source_page_end': 4,
                    },
                    {
                        'plot_goal': 'Follow contradictions toward river lead',
                        'mandatory_events': ['Spot conflicting clue', 'Choose route'],
                        'npc': ['Vendor'],
                        'locations': ['Square Exit'],
                        'source_page_start': 5,
                        'source_page_end': 5,
                    },
                ],
            },
            ensure_ascii=False,
        )

    return '{}'


def _mock_llm_teaser_boundary(prompt: str) -> str:
    if 'TASK: IDENTIFY_SCRIPT_STRUCTURE_LABELS' in prompt:
        labels = []
        for page_no in _pages_from_prompt(prompt):
            if page_no == 1:
                labels.append({'page': page_no, 'label': 'story', 'confidence': 0.56})
            elif page_no <= 4:
                labels.append({'page': page_no, 'label': 'front', 'confidence': 0.95})
            elif page_no <= 9:
                labels.append({'page': page_no, 'label': 'story', 'confidence': 0.91})
            else:
                labels.append({'page': page_no, 'label': 'appendix', 'confidence': 0.93})
        return json.dumps({'page_labels': labels}, ensure_ascii=False)

    if 'TASK: IDENTIFY_SCRIPT_STRUCTURE' in prompt:
        return json.dumps({'story': {'start_page': 5, 'end_page': 9}}, ensure_ascii=False)

    if 'TASK: EXTRACT_KNOWLEDGE_ONLY' in prompt:
        if 'Phase: front_knowledge' in prompt:
            return json.dumps(
                {
                    'knowledge': [
                        {
                            'knowledge_type': 'background',
                            'title': 'Front Matter',
                            'content': 'Campaign setup before main play.',
                            'source_page_start': 2,
                            'source_page_end': 4,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                'knowledge': [
                    {
                        'knowledge_type': 'rule',
                        'title': 'Appendix Note',
                        'content': 'Post-story optional material.',
                        'source_page_start': 10,
                        'source_page_end': 11,
                    }
                ]
            },
            ensure_ascii=False,
        )

    if 'TASK: EXTRACT_STORY_SCENES' in prompt:
        return json.dumps(
            {
                'scenes': [
                    {
                        'scene_goal': 'Start the true adventure',
                        'scene_description': 'Players enter the actionable story segment and must choose whom to trust. Conflicts escalate over several beats, and each choice changes immediate risks.',
                        'scene_summary': 'Main story block.',
                        'scene_type': 'normal',
                        'source_page_start': 5,
                        'source_page_end': 9,
                        'plots': [
                            {
                                'plot_goal': 'Initial confrontation and information check',
                                'mandatory_events': ['Meet faction contact'],
                                'npc': ['Contact'],
                                'locations': ['Market'],
                                'source_page_start': 5,
                                'source_page_end': 7,
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )

    if 'TASK: REFINE_SCENE_PLOTS' in prompt:
        return json.dumps(
            {
                'scene_type': 'normal',
                'plots': [
                    {
                        'plot_goal': 'Align with one side under uncertainty',
                        'mandatory_events': ['Choose ally'],
                        'npc': ['Contact'],
                        'locations': ['Market'],
                        'source_page_start': 5,
                        'source_page_end': 7,
                    },
                    {
                        'plot_goal': 'Execute the follow-up move and trigger consequences',
                        'mandatory_events': ['Act on chosen lead'],
                        'npc': ['Rival'],
                        'locations': ['Warehouse'],
                        'source_page_start': 8,
                        'source_page_end': 9,
                    },
                ],
            },
            ensure_ascii=False,
        )

    return '{}'


def main() -> int:
    mock_pages = [
        'Preface setup and table usage notes.',
        'NPC profiles and hidden truth notes.',
        'Story begins in town square.',
        'Witnesses disagree on suspect details.',
        'Players decide where to investigate next.',
        'Transition to the river road.',
        'Appendix: optional rule variants.',
        'Appendix: enemy stat blocks.',
    ]

    print('[test_parser] input pages:')
    for i, p in enumerate(mock_pages, start=1):
        print(f'  page{i}: {p}')

    try:
        bundle = parse_script_bundle(mock_pages, pages_per_scene=4, llm_client=_mock_llm)
        scenes = bundle.get('scenes', [])
        knowledge = bundle.get('knowledge', [])
        structure = bundle.get('structure', {})

        print('[test_parser] output bundle:')
        print('  structure ->', structure)
        print('  scenes ->', scenes)
        print('  knowledge ->', knowledge)

        assert structure.get('story') == {'start_page': 3, 'end_page': 6}
        assert structure.get('front_knowledge') == {'start_page': 1, 'end_page': 2}
        assert structure.get('appendix_knowledge') == {'start_page': 7, 'end_page': 8}

        manual_bundle = parse_script_bundle(
            mock_pages,
            pages_per_scene=4,
            llm_client=_mock_llm,
            story_start_page=4,
            story_end_page=6,
        )
        manual_structure = manual_bundle.get('structure', {})
        manual_warnings = manual_bundle.get('warnings', [])
        assert manual_structure.get('front_knowledge') == {'start_page': 1, 'end_page': 3}
        assert manual_structure.get('story') == {'start_page': 4, 'end_page': 6}
        assert manual_structure.get('appendix_knowledge') == {'start_page': 7, 'end_page': 8}
        assert any('manual story range applied (4-6)' in w for w in manual_warnings)

        assert len(knowledge) == 3, 'knowledge extraction count mismatch'
        for item in knowledge:
            s = item.get('source_page_start', 0)
            e = item.get('source_page_end', 0)
            assert not (3 <= s <= 6 and 3 <= e <= 6), 'story pages should not be stored as knowledge'

        assert len(scenes) == 2, 'scene extraction count mismatch'
        assert scenes[0]['scene_id'] == 'scene_1'
        assert scenes[0].get('source_page_start') == 3
        assert scenes[0].get('source_page_end') == 5
        assert len(scenes[0].get('plots', [])) >= 2, 'normal scene should have at least two plots after refinement'

        assert scenes[1].get('scene_type') == 'transition', 'transition scene type should be preserved'
        assert len(scenes[1].get('plots', [])) == 1, 'transition scene can keep one plot'

        compatibility_scenes = parse_script(mock_pages, pages_per_scene=4, llm_client=_mock_llm)
        assert len(compatibility_scenes) == 2, 'parse_script compatibility wrapper failed'

        teaser_pages = [
            'Dramatic teaser line that hints at danger.',
            'Module setup and constraints.',
            'Background timeline and true culprit notes.',
            'NPC profiles and relationship map.',
            'Story scene starts with playable choices.',
            'Investigation escalates with branching clues.',
            'Confrontation with first antagonist.',
            'Tactical pivot after failed negotiation.',
            'Scene closes with next objective set.',
            'Appendix rules and optional variants.',
            'Appendix stat blocks and references.',
        ]
        bundle2 = parse_script_bundle(teaser_pages, pages_per_scene=5, llm_client=_mock_llm_teaser_boundary)
        structure2 = bundle2.get('structure', {})
        scenes2 = bundle2.get('scenes', [])
        knowledge2 = bundle2.get('knowledge', [])

        assert structure2.get('front_knowledge') == {'start_page': 1, 'end_page': 4}
        assert structure2.get('story') == {'start_page': 5, 'end_page': 9}
        assert structure2.get('appendix_knowledge') == {'start_page': 10, 'end_page': 11}

        assert scenes2, 'teaser case should still produce story scenes'
        for scene in scenes2:
            s = scene.get('source_page_start', 0)
            e = scene.get('source_page_end', 0)
            assert 5 <= s <= 9 and 5 <= e <= 9, 'teaser case scene should stay inside story range'

        for item in knowledge2:
            s = item.get('source_page_start', 0)
            e = item.get('source_page_end', 0)
            assert e <= 4 or s >= 10, 'teaser case knowledge should stay outside story range'

        print('[test_parser] result: PASS')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'[test_parser] result: FAIL -> {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
