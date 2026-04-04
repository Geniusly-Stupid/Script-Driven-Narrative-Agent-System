import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser import detect_source_type, parse_script, parse_script_bundle, read_uploaded_document


def _pages_from_prompt(prompt: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"\[PAGE\s+(\d+)\]", prompt)]


def _extract_scene_heading(prompt: str) -> str:
    match = re.search(r"Scene Heading:\s*(.+)", prompt)
    return match.group(1).strip() if match else "Scene"


def _count_plots_in_metadata_prompt(prompt: str) -> int:
    return len(re.findall(r"(?m)^Plot \d+$", prompt))


def _line_slice(text: str, start: int, end: int) -> str:
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def _assert_contiguous_spans(rows: list[dict], start_key: str, end_key: str, expected_start: int, expected_end: int, label: str):
    previous_end = expected_start - 1
    for row in rows:
        start = int(row[start_key])
        end = int(row[end_key])
        assert start == previous_end + 1, f"{label} gap/overlap near {row}"
        assert end >= start, f"{label} inverted span near {row}"
        previous_end = end
    assert previous_end == expected_end, f"{label} does not cover expected end"


def _mock_pdf_llm(prompt: str) -> str:
    if "TASK: IDENTIFY_SCRIPT_STRUCTURE" in prompt:
        return json.dumps({"story": {"start_page": 3, "end_page": 6}}, ensure_ascii=False)

    if "TASK: EXTRACT_KNOWLEDGE_ONLY" in prompt:
        if "Phase: front_knowledge" in prompt:
            return json.dumps(
                {
                    "knowledge": [
                        {
                            "knowledge_type": "setting",
                            "title": "Module Setup",
                            "content": "Town setup and campaign constraints.",
                            "source_page_start": 1,
                            "source_page_end": 1,
                        },
                        {
                            "knowledge_type": "npc",
                            "title": "Important NPC",
                            "content": "Key motivations for preface NPC.",
                            "source_page_start": 2,
                            "source_page_end": 2,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "knowledge": [
                    {
                        "knowledge_type": "rule",
                        "title": "Appendix Rule",
                        "content": "Optional sanity variant at the end.",
                        "source_page_start": 7,
                        "source_page_end": 8,
                    }
                ]
            },
            ensure_ascii=False,
        )

    if "TASK: EXTRACT_STORY_SCENES" in prompt:
        return json.dumps(
            {
                "scenes": [
                    {
                        "scene_goal": "Investigate first incident",
                        "scene_description": "The party reaches the square and interviews witnesses. Conflicting accounts push them toward the river and reveal faction pressure. They need to decide whether to trust local militia or follow unofficial clues.",
                        "scene_summary": "Opening investigation beat.",
                        "scene_type": "normal",
                        "source_page_start": 3,
                        "source_page_end": 5,
                        "plots": [
                            {
                                "plot_goal": "Collect witness account",
                                "mandatory_events": ["Interview witness"],
                                "npc": ["Militia Liaison"],
                                "locations": ["Town Square"],
                                "source_page_start": 3,
                                "source_page_end": 5,
                            }
                        ],
                    },
                    {
                        "scene_goal": "Move to next location",
                        "scene_description": "The team departs quickly before nightfall.",
                        "scene_summary": "Bridge transition.",
                        "scene_type": "transition",
                        "source_page_start": 6,
                        "source_page_end": 6,
                        "plots": [
                            {
                                "plot_goal": "Travel to river road",
                                "mandatory_events": ["Leave square"],
                                "npc": [],
                                "locations": ["Road"],
                                "source_page_start": 6,
                                "source_page_end": 6,
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        )

    if "TASK: REFINE_SCENE_PLOTS" in prompt:
        return json.dumps(
            {
                "scene_type": "normal",
                "plots": [
                    {
                        "plot_goal": "Interview locals and collect baseline testimony",
                        "mandatory_events": ["Interview witness", "Cross-check statements"],
                        "npc": ["Militia Liaison", "Vendor"],
                        "locations": ["Town Square"],
                        "source_page_start": 3,
                        "source_page_end": 4,
                    },
                    {
                        "plot_goal": "Follow contradictions toward river lead",
                        "mandatory_events": ["Spot conflicting clue", "Choose route"],
                        "npc": ["Vendor"],
                        "locations": ["Square Exit"],
                        "source_page_start": 5,
                        "source_page_end": 5,
                    },
                ],
            },
            ensure_ascii=False,
        )

    return "{}"


def _mock_markdown_llm(prompt: str) -> str:
    if "TASK: EXTRACT_KNOWLEDGE_ONLY" in prompt:
        return json.dumps(
            {
                "knowledge": [
                    {
                        "knowledge_type": "background",
                        "title": "Keeper Information",
                        "content": "Background lore before play starts.",
                        "source_page_start": 3,
                        "source_page_end": 4,
                    }
                ]
            },
            ensure_ascii=False,
        )

    if "TASK: CLASSIFY_MARKDOWN_SCENE_CANDIDATES" in prompt:
        count = len(re.findall(r"(?m)^Candidate \d+$", prompt))
        return json.dumps(
            {
                "decisions": [
                    {"candidate_index": 1, "action": "keep"},
                    *[
                        {"candidate_index": idx, "action": "merge_with_previous"}
                        for idx in range(2, count + 1)
                    ],
                ]
            },
            ensure_ascii=False,
        )

    if "TASK: CLASSIFY_MARKDOWN_PLOT_BLOCKS" in prompt:
        if "Scene Heading: ASKING AROUND THE NEIGHBORHOOD" in prompt:
            return json.dumps(
                {
                    "decisions": [
                        {"block_index": 1, "role": "plot", "attach_to": ""},
                        {"block_index": 2, "role": "auxiliary", "attach_to": "previous"},
                    ]
                },
                ensure_ascii=False,
            )
        if "Scene Heading: TALKING TO THE POLICE" in prompt:
            return json.dumps(
                {
                    "decisions": [
                        {"block_index": 1, "role": "plot", "attach_to": ""},
                        {"block_index": 2, "role": "plot", "attach_to": ""},
                        {"block_index": 3, "role": "plot", "attach_to": ""},
                    ]
                },
                ensure_ascii=False,
            )
        count = len(re.findall(r"(?m)^Block \d+$", prompt))
        return json.dumps(
            {
                "decisions": [
                    {"block_index": 1, "role": "plot", "attach_to": ""},
                    *[
                        {"block_index": idx, "role": "plot", "attach_to": ""}
                        for idx in range(2, count + 1)
                    ],
                ]
            },
            ensure_ascii=False,
        )

    if "TASK: EXTRACT_FIXED_SCENE_METADATA" in prompt:
        scene_heading = _extract_scene_heading(prompt)
        plot_count = _count_plots_in_metadata_prompt(prompt)
        return json.dumps(
            {
                "scene_goal": scene_heading.title(),
                "scene_description": f"{scene_heading.title()} keeps the playable material together without changing the fixed spans.",
                "scene_summary": f"{scene_heading.title()} summary.",
                "scene_type": "transition" if scene_heading.lower() == "conclusion" else "normal",
                "plots": [
                    {
                        "plot_index": idx,
                        "plot_goal": f"{scene_heading.title()} plot {idx}",
                        "mandatory_events": [f"Event {idx}"],
                        "npc": [],
                        "locations": [],
                    }
                    for idx in range(1, plot_count + 1)
                ],
            },
            ensure_ascii=False,
        )

    return "{}"


def main() -> int:
    mock_pages = [
        "Preface setup and table usage notes.",
        "NPC profiles and hidden truth notes.",
        "Story begins in town square.",
        "Witnesses disagree on suspect details.",
        "Players decide where to investigate next.",
        "Transition to the river road.",
        "Appendix: optional rule variants.",
        "Appendix: enemy stat blocks.",
    ]

    markdown_text = "\n".join(
        [
            "# PAPER CHASE",
            "",
            "### KEEPER INFORMATION",
            "Background lore before play starts.",
            "",
            "### START",
            "Opening lead.",
            "Opening follow-up.",
            "",
            "#### ASKING AROUND THE NEIGHBORHOOD",
            "Neighbors point toward the cemetery.",
            "",
            "##### TWO INVESTIGATORS",
            "Keeper note for scaling the scene.",
            "",
            "#### TALKING TO THE POLICE",
            "The police are cautious but talkative.",
            "",
            "##### About Burglaries in the Area",
            "Reports connect the thefts to the missing books.",
            "",
            "##### Asking About the Cemetery",
            "An officer shares rumors about late-night visits.",
            "",
            "### CONCLUSION",
            "The investigator chooses the next move.",
        ]
    )

    print("[test_parser] input pages:")
    for idx, page in enumerate(mock_pages, start=1):
        print(f"  page{idx}: {page}")

    try:
        assert detect_source_type("scenario.pdf", "application/pdf") == "pdf"
        assert detect_source_type("scenario.markdown", "text/markdown") == "markdown"
        try:
            detect_source_type("scenario.txt", "text/plain")
            raise AssertionError("unsupported file type should raise ValueError")
        except ValueError:
            pass

        bundle = parse_script_bundle(mock_pages, pages_per_scene=4, llm_client=_mock_pdf_llm)
        structure = bundle.get("structure", {})
        scenes = bundle.get("scenes", [])
        knowledge = bundle.get("knowledge", [])

        assert structure.get("story") == {"start_page": 3, "end_page": 6}
        assert structure.get("front_knowledge") == {"start_page": 1, "end_page": 2}
        assert structure.get("appendix_knowledge") == {"start_page": 7, "end_page": 8}
        assert len(knowledge) == 3, "knowledge extraction count mismatch"
        assert len(scenes) == 2, "scene extraction count mismatch"
        assert len(scenes[0]["plots"]) == 2, "normal pdf scene should still refine into two plots"
        assert all(plot.get("raw_text") for scene in scenes for plot in scene.get("plots", [])), "pdf plots should store raw_text"

        manual_bundle = parse_script_bundle(
            mock_pages,
            pages_per_scene=4,
            llm_client=_mock_pdf_llm,
            story_start_page=4,
            story_end_page=6,
        )
        manual_structure = manual_bundle.get("structure", {})
        assert manual_structure.get("front_knowledge") == {"start_page": 1, "end_page": 3}
        assert manual_structure.get("story") == {"start_page": 4, "end_page": 6}
        assert manual_structure.get("appendix_knowledge") == {"start_page": 7, "end_page": 8}

        compatibility_scenes = parse_script(mock_pages, pages_per_scene=4, llm_client=_mock_pdf_llm)
        assert len(compatibility_scenes) == 2, "parse_script compatibility wrapper failed"

        cropped_doc = read_uploaded_document(
            "paper_chase.md",
            markdown_text.encode("utf-8"),
            start_unit=3,
            end_unit=24,
            mime_type="text/markdown",
        )
        assert cropped_doc.source_type == "markdown"
        assert cropped_doc.unit_label == "line"
        assert cropped_doc.display_start == 3 and cropped_doc.display_end == 24
        assert cropped_doc.outline, "markdown outline should capture headings"
        assert all("<img" not in segment.text.lower() for segment in cropped_doc.segments), "decorative image html should be removed"

        markdown_doc = read_uploaded_document(
            "paper_chase.md",
            markdown_text.encode("utf-8"),
            mime_type="text/markdown",
        )
        markdown_bundle = parse_script_bundle(
            source_document=markdown_doc,
            llm_client=_mock_markdown_llm,
            story_start_page=6,
            story_end_page=26,
        )

        markdown_structure = markdown_bundle.get("structure", {})
        markdown_scenes = markdown_bundle.get("scenes", [])
        markdown_knowledge = markdown_bundle.get("knowledge", [])
        markdown_source_meta = markdown_bundle.get("source_metadata", {})

        assert markdown_structure.get("front_knowledge") == {"start_page": 1, "end_page": 5}
        assert markdown_structure.get("story") == {"start_page": 6, "end_page": 26}
        assert markdown_structure.get("appendix_knowledge") is None
        assert markdown_source_meta.get("source_type") == "markdown"
        assert markdown_source_meta.get("source_unit_label") == "line"
        assert len(markdown_knowledge) == 1, "front matter should remain knowledge"

        assert len(markdown_scenes) == 4, "markdown headings should yield four contiguous scenes"
        _assert_contiguous_spans(markdown_scenes, "source_page_start", "source_page_end", 6, 26, "scene")

        expected_scene_spans = [(6, 9), (10, 15), (16, 24), (25, 26)]
        for scene, expected in zip(markdown_scenes, expected_scene_spans, strict=True):
            assert (scene["source_page_start"], scene["source_page_end"]) == expected
            _assert_contiguous_spans(
                scene["plots"],
                "source_page_start",
                "source_page_end",
                expected[0],
                expected[1],
                f"plot in {scene['scene_id']}",
            )
            for plot in scene["plots"]:
                assert plot["source_page_start"] >= scene["source_page_start"]
                assert plot["source_page_end"] <= scene["source_page_end"]
                assert plot.get("raw_text"), "markdown plots should keep raw_text"

        assert len(markdown_scenes[0]["plots"]) == 1, "scene without level-5 headings should stay as one plot"
        assert len(markdown_scenes[1]["plots"]) == 1, "auxiliary-only level-5 heading should merge back into one plot"
        assert len(markdown_scenes[2]["plots"]) == 3, "scene intro plus substantive level-5 headings should become parallel plots"
        assert len(markdown_scenes[3]["plots"]) == 1, "conclusion may remain a single plot"

        expected_talking_plot_1 = _line_slice(markdown_text, 16, 18)
        expected_talking_plot_2 = _line_slice(markdown_text, 19, 21)
        expected_talking_plot_3 = _line_slice(markdown_text, 22, 24)
        assert markdown_scenes[2]["plots"][0]["raw_text"] == expected_talking_plot_1
        assert markdown_scenes[2]["plots"][1]["raw_text"] == expected_talking_plot_2
        assert markdown_scenes[2]["plots"][2]["raw_text"] == expected_talking_plot_3

        print("[test_parser] result: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[test_parser] result: FAIL -> {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
