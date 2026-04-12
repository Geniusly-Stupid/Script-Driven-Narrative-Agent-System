import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser import detect_source_type, parse_script, parse_script_bundle, read_uploaded_document


def _line_slice(text: str, start: int, end: int) -> str:
    lines = text.split("\n")
    return "\n".join(lines[start - 1 : end])


def main() -> int:
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

    markdown_with_decorative_html = "\n".join(
        [
            "# PAPER CHASE",
            "",
            "<div><img src='cover.png' /></div>",
            "",
            "### START",
            "Playable text.",
        ]
    )

    prompts: list[str] = []

    def mock_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "You are summarizing a narrative script." in prompt:
            assert "New input (scene & plots):" in prompt
            return "The investigation moves through several connected scenes as the player follows leads and interviews sources. Key actions include asking around, talking to the police, and pursuing multiple lines of inquiry. Important discoveries point toward the cemetery, the missing books, and Douglas Kimball's disappearance."

        assert "You are an information extraction assistant." in prompt
        assert "Return only JSON. Do not include explanations." in prompt
        assert "--- BEGIN INPUT ---" in prompt
        assert "--- END INPUT ---" in prompt

        knowledge = []
        if "Background lore before play starts." in prompt:
            knowledge = [
                {
                    "knowledge_type": "setting",
                    "title": "Keeper Information",
                    "content": "Background lore before play starts.",
                }
            ]

        return json.dumps(
            {
                "scene": {
                    "scene_name": "Extracted Scene",
                    "scene_goal": "Advance the investigation",
                    "scene_description": "The player investigates the current location and follows the available leads.",
                },
                "plots": [
                    {
                        "plot_name": "Primary lead",
                        "plot_goal": "Advance the investigation",
                    },
                    {
                        "plot_name": "Secondary lead",
                        "plot_goal": "Gather more context",
                    },
                    {
                        "plot_name": "Third lead",
                        "plot_goal": "Confirm the clue",
                    },
                ],
                "knowledge": knowledge,
            },
            ensure_ascii=False,
        )

    try:
        assert detect_source_type("scenario.markdown", "text/markdown") == "markdown"
        assert detect_source_type("scenario.md") == "markdown"
        for file_name, mime_type in [("scenario.pdf", "application/pdf"), ("scenario.txt", "text/plain")]:
            try:
                detect_source_type(file_name, mime_type)
                raise AssertionError("unsupported file type should raise ValueError")
            except ValueError:
                pass

        cropped_doc = read_uploaded_document("paper_chase.md", markdown_with_decorative_html.encode("utf-8"), mime_type="text/markdown")
        assert cropped_doc.source_type == "markdown"
        assert cropped_doc.outline
        assert all("<img" not in segment.text.lower() for segment in cropped_doc.segments)

        markdown_doc = read_uploaded_document("paper_chase.md", markdown_text.encode("utf-8"), mime_type="text/markdown")
        bundle = parse_script_bundle(source_document=markdown_doc, llm_client=mock_llm)

        scenes = bundle.get("scenes", [])
        knowledge = bundle.get("knowledge", [])
        script_summary = bundle.get("script_summary", "")
        source_meta = bundle.get("source_metadata", {})

        assert source_meta.get("source_type") == "markdown"
        assert source_meta.get("line_count") == len(markdown_text.split("\n"))
        assert len(knowledge) == 1
        assert "The investigation moves through several connected scenes" in script_summary

        assert len(scenes) == 5
        assert [scene.get("scene_name", "") for scene in scenes] == ["Extracted Scene"] * 5
        assert [len(scene["plots"]) for scene in scenes] == [1, 1, 1, 3, 1]
        assert all(plot.get("raw_text") for scene in scenes for plot in scene.get("plots", []))
        assert scenes[0]["scene_index"] == 1 and scenes[-1]["scene_index"] == 5
        assert scenes[3]["plots"][0]["plot_name"] == "Primary lead"
        assert scenes[3]["plots"][1]["plot_name"] == "Secondary lead"
        assert scenes[3]["plots"][2]["plot_name"] == "Third lead"

        assert scenes[3]["plots"][0]["raw_text"] == _line_slice(markdown_text, 16, 18)
        assert scenes[3]["plots"][1]["raw_text"] == _line_slice(markdown_text, 19, 21)
        assert scenes[3]["plots"][2]["raw_text"] == _line_slice(markdown_text, 22, 24)

        compatibility_scenes = parse_script([markdown_text], llm_client=mock_llm)
        assert len(compatibility_scenes) == 5

        long_prompts: list[str] = []

        def truncation_llm(prompt: str) -> str:
            long_prompts.append(prompt)
            if "You are summarizing a narrative script." in prompt:
                return "A long scene summary."
            return json.dumps(
                {
                    "scene": {
                        "scene_name": "Long Scene",
                        "scene_goal": "Stay within the token budget",
                        "scene_description": "A long scene used to test prompt truncation.",
                    },
                    "plots": [
                        {
                            "plot_name": "Long plot",
                            "plot_goal": "Check truncation",
                        }
                    ],
                    "knowledge": [],
                }
            )

        long_doc = read_uploaded_document(
            "long_scene.md",
            ("### START\n" + ("A" * 12050)).encode("utf-8"),
            mime_type="text/markdown",
        )
        parse_script_bundle(source_document=long_doc, llm_client=truncation_llm)
        long_content = long_prompts[0].split("--- BEGIN INPUT ---\n", 1)[1].split("\n--- END INPUT ---", 1)[0]
        assert len(long_content) == 10000

        empty_bundle = parse_script_bundle(
            source_document=markdown_doc,
            llm_client=lambda prompt: (
                ""
                if "You are summarizing a narrative script." in prompt
                else json.dumps(
                    {
                        "scene": {},
                        "plots": [],
                        "knowledge": [
                            {
                                "knowledge_type": "setting",
                                "title": "Background",
                                "content": "Knowledge-only content.",
                            }
                        ],
                    }
                )
            ),
        )
        assert empty_bundle["scenes"] == []
        assert len(empty_bundle["knowledge"]) == 5
        assert empty_bundle["script_summary"] == ""

        print("[test_parser] result: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[test_parser] result: FAIL -> {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
