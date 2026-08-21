"""Maintenance-only extractor for the immutable F0A compact goldens.

This script is never imported or invoked by pytest. Sources and destination are
explicit, sources are opened read-only, and an existing destination is rejected
unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tests.support.provenance import relative_hashes, sha256_file


SCHEMA_VERSION = "1.0"
BASELINE_COMMIT = "806832419b0ab3ac40050d8c05c3bd0bed5098f6"
FAMILIES = ("T3", "Q4", "Q8")


def read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = stream.readline().strip()
    names = header.split(",")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    table = np.atleast_1d(table)
    return names, {name: np.asarray(table[name]) for name in names}


def read_table(path: Path, *, dtype: type[np.floating[Any]] | type[np.integer[Any]]) -> np.ndarray:
    table = np.loadtxt(path, delimiter=",", comments="#", dtype=dtype)
    return np.atleast_2d(table)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_artifacts(root: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }


def extract_subset(repository: Path, source: Path, output: Path) -> None:
    displacement_path = source / "displacements.csv"
    strain_path = source / "strain.csv"
    stats_path = source / "stats.json"
    paths = [displacement_path, strain_path, stats_path]
    displacement_header, displacement = read_csv(displacement_path)
    strain_header, strain = read_csv(strain_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    arrays: dict[str, np.ndarray] = {"schema_version": np.array(SCHEMA_VERSION)}
    arrays.update({f"displacement__{name}": value for name, value in displacement.items()})
    arrays.update({f"strain__{name}": value for name, value in strain.items()})
    output.mkdir(parents=True)
    np.savez_compressed(output / "baseline.npz", **arrays)
    write_json(
        output / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline_kind": "subset_ring_full_artifact",
            "headers": {"displacements": displacement_header, "strain": strain_header},
            "row_counts": {
                "displacements": len(displacement[displacement_header[0]]),
                "strain": len(strain[strain_header[0]]),
            },
            "statistics": stats,
            "source_artifacts": source_artifacts(repository, paths),
        },
    )
    image_paths = [repository / "case/mono_DIC/ring" / name for name in ("001.bmp", "002.bmp", "003.bmp")]
    write_json(
        output / "provenance.json",
        provenance(
            repository,
            source_case="case/mono_DIC/ring",
            artifact_paths=paths,
            config_paths=[repository / "config/subset_2d.yaml", repository / "config/case_paths.yaml"],
            input_paths=image_paths,
            input_order=[
                {"path": "case/mono_DIC/ring/001.bmp", "role": "reference"},
                {"path": "case/mono_DIC/ring/002.bmp", "role": "deformed"},
                {"path": "case/mono_DIC/ring/003.bmp", "role": "roi"},
            ],
        ),
    )


def extract_mesh(repository: Path, source: Path, output: Path) -> None:
    output.mkdir(parents=True)
    artifact_paths = [source / "mesh_generation_summary.json"]
    family_summaries: dict[str, Any] = {}
    for family in FAMILIES:
        source_dir = source / family
        target_dir = output / family
        target_dir.mkdir()
        nodes_path = source_dir / f"nodes_{family}.txt"
        elements_path = source_dir / f"elements_{family}.txt"
        final_path = source_dir / "final_U.csv"
        dense_path = source_dir / "dense_U.csv"
        strain_path = source_dir / "dense_strain.csv"
        summary_path = source_dir / "summary.json"
        family_paths = [nodes_path, elements_path, final_path, dense_path, strain_path, summary_path]
        artifact_paths.extend(family_paths)

        nodes = read_table(nodes_path, dtype=np.float64)
        elements = read_table(elements_path, dtype=np.int64)
        final_header, final = read_csv(final_path)
        dense_header, dense = read_csv(dense_path)
        strain_header, strain = read_csv(strain_path)
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.array(SCHEMA_VERSION),
            "nodes": nodes,
            "elements": elements,
        }
        arrays.update({f"final_U__{name}": value for name, value in final.items()})
        arrays.update({f"dense_U__{name}": value for name, value in dense.items()})
        arrays.update({f"dense_strain__{name}": value for name, value in strain.items()})
        np.savez_compressed(target_dir / "baseline.npz", **arrays)
        family_summary = {
            "schema_version": SCHEMA_VERSION,
            "baseline_kind": "mesh_full_artifact",
            "element_family": family,
            "topology": {
                "node_count": int(nodes.shape[0]),
                "element_count": int(elements.shape[0]),
                "nodes_columns": int(nodes.shape[1]),
                "elements_columns": int(elements.shape[1]),
            },
            "headers": {
                "final_U": final_header,
                "dense_U": dense_header,
                "dense_strain": strain_header,
            },
            "row_counts": {
                "final_U": len(final[final_header[0]]),
                "dense_U": len(dense[dense_header[0]]),
                "dense_strain": len(strain[strain_header[0]]),
            },
            "statistics": source_summary,
            "source_artifacts": source_artifacts(repository, family_paths),
        }
        family_summaries[family] = family_summary
        write_json(target_dir / "summary.json", family_summary)

    images = sorted((repository / "case/mono_DIC/01").glob("*.bmp"))
    input_order = []
    for index, path in enumerate(images):
        role = "reference" if index == 0 else "roi" if index == len(images) - 1 else "deformed"
        input_order.append({"path": str(path.relative_to(repository)), "role": role})
    write_json(
        output / "provenance.json",
        provenance(
            repository,
            source_case="case/mono_DIC/01",
            artifact_paths=artifact_paths,
            config_paths=[repository / "config/mesh_2d.yaml", repository / "config/case_paths.yaml"],
            input_paths=images,
            input_order=input_order,
        ),
    )


def provenance(
    repository: Path,
    *,
    source_case: str,
    artifact_paths: list[Path],
    config_paths: list[Path],
    input_paths: list[Path],
    input_order: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "repository": "Traditional-DIC",
        "source_case": source_case,
        "source_artifacts": source_artifacts(repository, artifact_paths),
        "input_order": input_order,
        "input_sha256": relative_hashes(repository, input_paths),
        "config_paths": [str(path.relative_to(repository)) for path in config_paths],
        "config_sha256": relative_hashes(repository, config_paths),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment": {
            "extraction": "conda environment tradic",
            "historical_solver_build": "NOT_RECORDED",
            "historical_python": "NOT_RECORDED",
            "historical_platform": "NOT_RECORDED",
        },
        "extraction_method": "tests.support.extract_f0_baseline full-artifact compressed NPZ",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--subset-source", type=Path, required=True)
    parser.add_argument("--mesh-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            parser.error(f"output already exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    extract_subset(repository, args.subset_source.resolve(), output / "subset_ring")
    extract_mesh(repository, args.mesh_source.resolve(), output / "mesh_case01")
    generated = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline_commit": BASELINE_COMMIT,
            "baseline_scope": "F0A subset + mesh behavior artifacts",
            "immutable_at_pytest_runtime": True,
            "datasets": {
                "subset_ring": "subset_ring",
                "mesh_case01": "mesh_case01",
            },
            "files": {
                str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in generated
            },
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
