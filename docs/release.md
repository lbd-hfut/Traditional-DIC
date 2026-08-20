# Release qualification

R2 qualifies, but does not publish, Traditional-DIC distributions.  The
supported release target is Linux x86-64 with CPython 3.11 and the Conda-native
runtime described in [`environment.yml`](../environment.yml): OpenCV 5, Ceres,
and yaml-cpp.

`Source regression` runs on pull requests and `main`.  `Release qualification`
is manually dispatched and builds an sdist and wheel, uploads the exact dist
artifact, then installs and qualifies that artifact in a separate job.  It
checks installed CLI, MCP STDIO, localhost Streamable HTTP, one Subset run, one
T3 Mesh run, F4 metadata, and the 3D Mesh restrictions.

The workflow has read-only repository permissions and performs no publication.
After review, R3 may publish only the exact SHA-qualified artifacts; version
bumps and tags such as `v0.1.0` require explicit approval.

For local reproduction, activate the supported Conda environment and run:

```bash
python -m pip install build scikit-build-core pybind11
python qualification/package/verify_release.py \
  --source-root "$PWD" \
  --build \
  --case-root "$PWD/case" \
  --work-root /tmp/traditional-dic-r2 \
  --report /tmp/traditional-dic-r2/r2-ci-qualification.json
```
