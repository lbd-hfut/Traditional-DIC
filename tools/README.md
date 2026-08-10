# Tools

Put standalone helper scripts here, such as environment checks, data conversion, batch processing, result inspection, and visualization helpers.

Tools may call public project APIs, but production code must not depend on this directory.

## Launching examples on Windows

Python 3.8+ does not search PATH for dependent DLLs, so `import traditional_dic`
fails unless the built runtime directories are registered via
`os.add_dll_directory`. Use the launcher to handle this:

```powershell
python tools/run_example.py examples/subset_2d.py --paths-config config\case_paths.yaml
```

It resolves the repo root from its own location, registers
`build/_opencv/runtime` and `build/_opencv/ucrt64/bin`, then runs the target
script with its arguments unchanged. `setup_dll_directories()` can be imported
from this module for reuse in other tools.

