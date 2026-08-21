# Traditional-DIC development guide

## Architecture and ownership

```text
config/case_paths.yaml ─┐
config/*.yaml ──────────┼→ F1 case.py + F2 config_resolver.py
                         ↓
                   F3 workflows/
                         ↓
              Python algorithm modules / pybind
                         ↓
                 C++ core: include/ + src/
                         ↓
                   F4 run_contract.py
                         ↓
                CLI and local MCP adapters
```

- `include/dic/`, `src/`: native scientific engine.
- `bindings/python/`: pybind module exposure; `_traditional_dic` is the extension.
- `python/traditional_dic/case.py`: deterministic F1 input discovery/validation.
- `python/traditional_dic/config_resolver.py`: F2 defaults, overlays, normalization, validation, provenance.
- `python/traditional_dic/workflows/`: F3 public orchestration facade.
- `python/traditional_dic/run_contract.py`: additive F4 metadata contract.
- `python/traditional_dic/cli.py`, `mcp_server.py`: adapters; keep semantics in shared contracts.
- `config/`: workflow and case configuration defaults.
- `tests/unit/`, `tests/regression/`: contracts and numerical/golden evidence.

## Local development environment

Use the repository's supported Conda environment:

```bash
conda env create -f environment.yml
conda activate traditional-dic
cmake -S . -B build -DTRADITIONAL_DIC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
PYTHONPATH=python python -m pytest tests -q
```

`PYTHONPATH=python` is a source-tree fallback only. Use installed wheel entry points for installed-mode checks.

## Change matrix

| Change | Likely files | Minimum validation | Full tests | Local package qualifier | Docs |
| --- | --- | --- | --- | --- | --- |
| Subset science | `src/subset`, bindings, `subset.py`, workflows/config | affected regression | yes | only if package boundary changes | user/development |
| Mesh science | `src/mesh`, bindings, `mesh.py`, workflows/config | mesh regressions | yes | only if package boundary changes | user/development |
| Stereo/Multiview | reconstruction/calibration/workflow modules/config | relevant 3D regressions | yes | only if package boundary changes | user/development/API |
| Case/config contracts | `case.py`, `config_resolver.py`, YAML | unit + resolver regressions | yes | no, unless resources/install changed | user/API/development |
| CLI/MCP adapter | `cli.py`, `mcp_server.py`, capabilities/tests | adapter tests | yes | MCP/package smoke if installed behavior changed | user/agent/MCP/SKILL |
| pybind/CMake/package | bindings, CMake, `pyproject.toml`, resources | rebuild/import | yes | yes | installation/development |
| docs only | Markdown and focused doc checks | link/API-import checks | recommended | no | affected documents |

## Scientific change policy

Scientific behavior must not change silently. A modification to an objective,
shape function, interpolation, iteration rule, threshold, regularization,
strain, reconstruction, outlier cleaning, or stitching requires:

1. Identify the affected regression evidence.
2. Run targeted tests and the full suite.
3. Explain and record intended numerical differences.
4. Update golden artifacts only with explicit scientific justification.

Never regenerate goldens simply because a scientific test fails.

## Contract maintenance

### Case and configuration

For a new case/config key, update the YAML default, resolver key checking and normalization, the workflow consumer, unit/regression coverage, and user/API documentation. Preserve fail-closed behavior and dotted override provenance.

### Workflow and F4

Public work belongs in `workflows/` and should accept `ResolvedCase` plus `ResolvedConfig`. Preserve `WorkflowRunResult` and all four F4 files. Do not turn an execution failure into a quality warning, or vice versa.

### CLI and MCP

Add shared Python contract behavior first; then adapt CLI and MCP. Avoid CLI-only semantics. The MCP registry is deliberately fixed at six public tools. A seventh tool is a public protocol change requiring server, capabilities, tests, `SKILL.md`, MCP/Agent documentation, and compatibility review. Prefer extending an existing semantic tool.

Changing Stereo or Multiview solver availability is an architectural/scientific review: both are currently Subset-only. Do not expose Mesh through a generic option.

### Native boundary

Python orchestration calls `traditional_dic._traditional_dic`, built from `bindings/python/module.cpp` and `bind_*.cpp`, which links the C++ core. Add native APIs in the correct header/source layer, bind them deliberately, rebuild in Conda, and avoid making raw binding symbols the normal user API without a facade and contract review.

## Tests and local qualification

The suite has 87 unit and 59 regression tests. Run all tests before integrating behavior changes:

```bash
PYTHONPATH=python python -m pytest tests/unit -v
PYTHONPATH=python python -m pytest tests/regression -v
PYTHONPATH=python python -m pytest tests -v
python -m compileall python/traditional_dic qualification tests
git diff --check
```

Run `qualification/package/verify_release.py` when changing `pyproject.toml`, CMake install rules, extension packaging, runtime YAML resources, entry points, MCP optional dependencies, environment/runtime integration, or installed-mode behavior. It is useful before handing a wheel to another local machine; it is not required after every scientific-only edit.

## Git and documentation practice

Use ordinary source branches such as `feature/...`, `fix/...`, or `experiment/...`; `main` is the integrated stable source. No hosted CI or release branch is required by this local-only project.

Update docs whenever a public case/config/workflow/F4/CLI/MCP/package contract changes. Keep [README](../README.md) concise, put operational detail in the user/agent guides, and put function signatures in the [API reference](api-reference.md).
