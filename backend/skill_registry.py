from functools import lru_cache
from pathlib import Path


SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _frontmatter(content):
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[1:index]
    return []


def _frontmatter_tools(content):
    tools = []
    in_tools = False
    for raw_line in _frontmatter(content):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("tools:"):
            in_tools = True
            inline = stripped.split(":", 1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                tools.extend(item.strip().strip("'\"") for item in inline[1:-1].split(",") if item.strip())
            continue
        if in_tools:
            if stripped.startswith("- "):
                tools.append(stripped[2:].strip().strip("'\""))
                continue
            if not raw_line.startswith((" ", "\t")):
                in_tools = False
    return [tool for tool in tools if tool]


@lru_cache(maxsize=1)
def load_skill_documents():
    documents = []
    if not SKILLS_DIR.exists():
        return documents
    for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        content = skill_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append({
            "name": skill_file.parent.name,
            "content": content,
            "tools": _frontmatter_tools(content),
        })
    return documents


def skills_prompt():
    blocks = []
    for skill in load_skill_documents():
        blocks.append(f"## Skill: {skill['name']}\n{skill['content']}")
    return "\n\n".join(blocks)


def skills_for_tools(tool_names):
    """Return skills whose frontmatter declares at least one selected tool."""
    selected = {str(name).strip() for name in (tool_names or []) if str(name).strip()}
    if not selected:
        return []
    return [
        skill
        for skill in load_skill_documents()
        if selected.intersection(skill.get("tools") or [])
    ]


def skills_prompt_for_tools(tool_names):
    """Render only the skills relevant to the selected tool names."""
    blocks = [
        f"## Skill: {skill['name']}\n{skill['content']}"
        for skill in skills_for_tools(tool_names)
    ]
    return "\n\n".join(blocks)


def skill_tool_names():
    names = []
    seen = set()
    for skill in load_skill_documents():
        for tool in skill.get("tools") or []:
            if tool not in seen:
                seen.add(tool)
                names.append(tool)
    return names
