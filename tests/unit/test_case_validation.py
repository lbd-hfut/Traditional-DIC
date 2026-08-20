import json
import os
from pathlib import Path

import pytest

from traditional_dic.case import inspect_case, resolve_case


pytestmark = pytest.mark.unit


def test_inspect_case_is_structured_and_fail_closed(tmp_path: Path) -> None:
    result = inspect_case("subset_2d", case_root=tmp_path)
    assert not result.valid
    assert result.errors[0].code == "INSUFFICIENT_IMAGES"
    assert result.to_dict()["errors"][0]["message"]


def test_serialization_is_json_stable(repository_root: Path) -> None:
    case = resolve_case("multiview_3d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    first = case.to_json(repository_root=repository_root)
    second = json.dumps(case.to_dict(repository_root=repository_root), indent=2, sort_keys=True) + "\n"
    assert first == second
    document = json.loads(first)
    assert not any(str(value).startswith("/home/") for value in _walk(document))


def test_resolution_is_cwd_independent(repository_root: Path, tmp_path: Path) -> None:
    original = Path.cwd()
    try:
        os.chdir(tmp_path)
        case = resolve_case("subset_2d", paths_config=repository_root / "config/case_paths.yaml", repository_root=repository_root)
    finally:
        os.chdir(original)
    assert case.case_root == repository_root / "case/mono_DIC/ring"
    assert case.frame("reference").path == repository_root / "case/mono_DIC/ring/001.bmp"


def _walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value
