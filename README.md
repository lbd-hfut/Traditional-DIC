# Traditional-DIC

Traditional-DIC is a local Digital Image Correlation toolkit. It combines a C++ scientific core with a Python workflow facade, a local command-line interface, and an optional local [Model Context Protocol (MCP)](docs/mcp.md) adapter.

It supports standalone 2D Subset-DIC and Mesh-DIC, plus Subset-based Stereo and Multiview 3D workflows. Computation, files, and MCP transports are local to the machine running the package.

## Supported workflows

| Workflow | CLI name | Python facade | MCP workflow | Solver contract |
| --- | --- | --- | --- | --- |
| 2D Subset-DIC | `subset-2d` | `run_subset_2d` | `subset_2d` | Subset |
| 2D Mesh-DIC | `mesh-2d` | `run_mesh_2d` | `mesh_2d` | Mesh (`T3`, `Q4`, `Q8`) |
| Stereo 3D-DIC | `stereo-3d` | `run_stereo_3d` | `stereo_3d` | **Subset only** |
| Multiview 3D-DIC | `multiview-3d` | `run_multiview_3d` | `multiview_3d` | **Subset only** |

Mesh is deliberately not available for Stereo or Multiview workflows. Ask the installed program for the machine-readable contract before automating a run:

```bash
traditional-dic capabilities --format json
```

## Architecture

```text
Human ── traditional-dic CLI ─┐
                               ├─ F1 case → F2 config → F3 workflow → F4 result metadata
Agent ── local MCP / STDIO ───┘                         ↓
                                                Python / pybind boundary
                                                        ↓
                                                C++ scientific core
                                                        ↓
                                              local CPU, files, outputs
```

The four F4 files written to every facade run are `manifest.json`, `status.json`, `metrics.json`, and `result.json`. They distinguish whether a run executed from the quality of its numerical result.

## Installation and platform support

The qualified deployment is **Linux x86-64, CPython 3.11, and the documented Conda native runtime**. It uses Conda-provided OpenCV 5, Ceres Solver, and yaml-cpp. This is not a manylinux, macOS, Windows, or arbitrary pip-only portability claim.

```bash
git clone <repository-url> Traditional-DIC
cd Traditional-DIC
conda env create -f environment.yml
conda activate traditional-dic
```

For users, build a wheel and install it into that environment; see the [installation guide](docs/installation.md). Installed commands do not need `PYTHONPATH` or a Git checkout.

## Quick start

The repository's `case/` directory contains source/test examples; it is not included in an installed wheel. Keep outputs outside input cases:

```bash
traditional-dic inspect --workflow subset-2d --case /path/to/case --format json
traditional-dic validate --workflow subset-2d --case /path/to/case --format json
traditional-dic run --workflow subset-2d --case /path/to/case \
  --output /tmp/traditional-dic-subset --format json
```

The recommended Python route is the same normalized contract:

```python
from traditional_dic.case import resolve_case
from traditional_dic.config_resolver import resolve_config
from traditional_dic.workflows import run_subset_2d

case = resolve_case("subset_2d", case_root="/path/to/case")
config = resolve_config("subset_2d")
run = run_subset_2d(case, config, output_root="/tmp/traditional-dic-subset")
print(run.status_path)
```

For an Agent, use the local MCP server and begin with `traditional_dic_capabilities`; the procedure is in the [Agent guide](docs/agent-guide.md).

## Documentation

- [Installation](docs/installation.md)
- [User guide](docs/user-guide.md)
- [Python API reference](docs/api-reference.md)
- [Agent guide](docs/agent-guide.md)
- [MCP transport](docs/mcp.md)
- [Development guide](docs/development.md)

## Development and testing

Source-tree development needs the compiled extension. From the repository root in the `traditional-dic` Conda environment:

```bash
cmake -S . -B build -DTRADITIONAL_DIC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
PYTHONPATH=python python -m pytest tests -q
```

The current suite contains 87 unit tests and 59 regression tests. See the [development guide](docs/development.md) for maintenance rules and local package qualification.

## Scope

Traditional-DIC is local scientific software. It does not provide Remote MCP, hosted computation, a web service, accounts, OAuth/JWT/RBAC, or a multi-user permission system.

## License

See [LICENSE](LICENSE).
