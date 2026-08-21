"""Low-maintenance guards for the repository's public documentation map."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from traditional_dic.mcp_server import mcp


def test_core_documentation_and_readme_links_exist(repository_root: Path) -> None:
    expected = (
        "README.md",
        "docs/installation.md",
        "docs/user-guide.md",
        "docs/agent-guide.md",
        "docs/development.md",
        "docs/api-reference.md",
        "docs/mcp.md",
    )
    for relative in expected:
        assert (repository_root / relative).is_file()

    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    for target in ("docs/installation.md", "docs/user-guide.md", "docs/agent-guide.md", "docs/development.md", "docs/api-reference.md", "docs/mcp.md"):
        assert target in readme


def test_agent_documentation_matches_the_six_tool_registry(repository_root: Path) -> None:
    agent_guide = (repository_root / "docs/agent-guide.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(traditional_dic_[a-z_]+)\(", agent_guide))
    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert documented == registered
