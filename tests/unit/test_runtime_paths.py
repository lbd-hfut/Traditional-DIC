from pathlib import Path

from traditional_dic._runtime import default_config_path, resolve_config_reference, runtime_root


def test_source_tree_runtime_paths_preserve_the_canonical_config_directory():
    root = runtime_root()
    expected = Path(__file__).resolve().parents[2] / "config" / "subset_2d.yaml"

    assert root == expected.parents[1]
    assert default_config_path("config/subset_2d.yaml") == expected
    assert resolve_config_reference("config/subset_2d.yaml", root) == expected
