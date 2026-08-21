# Installation

Traditional-DIC 0.1.0 is qualified on Linux/x86_64 with Python 3.11 and the
Conda-forge native runtime in [`environment.yml`](../environment.yml).  The
wheel contains the Python package, the pybind11 extension, and the default
workflow YAML files; it deliberately does not contain case datasets.

Create the supported runtime and install a built wheel:

```bash
conda env create -f environment.yml
conda activate traditional-dic
python -m pip install /path/to/traditional_dic-0.1.0-cp311-cp311-linux_x86_64.whl
traditional-dic capabilities --format json
```

The native extension links against the Conda-provided OpenCV 5, Ceres Solver,
and yaml-cpp libraries.  Activate this environment before running the CLI or
MCP server.  The qualified wheel tag is Linux CPython 3.11 only; this is not a
manylinux, macOS, or Windows portability claim.

To enable MCP, install the optional extra from the same artifact:

```bash
python -m pip install '/path/to/traditional_dic-0.1.0-cp311-cp311-linux_x86_64.whl[mcp]'
traditional-dic-mcp
```

The installed commands work from arbitrary directories.  Supply absolute case
paths and an explicit external output workspace, for example:

```bash
cd /tmp
traditional-dic inspect \
  --workflow subset-2d \
  --case /absolute/path/to/case \
  --format json
traditional-dic run \
  --workflow subset-2d \
  --case /absolute/path/to/case \
  --output /tmp/traditional-dic-run \
  --format json
```

`PYTHONPATH=python` is only a repository-development fallback.  It is not
needed for the installed distribution.
