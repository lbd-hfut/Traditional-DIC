# Agent Instructions

This repository uses this file as the root entry point for AI coding agents.

Before changing code, read and follow:

- `docs/engineering_handoff_zh.md`
- `README.md`
- `README_zh.md` when Chinese context is needed

Project hygiene rules:

- Build artifacts belong only in `build/`, `build_debug/`, `build_relwithdeb/`, `build_asan/`, or another `build*` directory.
- Test code and test scripts belong in `tests/`.
- Standalone helper scripts belong in `tools/`.
- Runtime, debug, and test logs belong in `logs/`.
- Do not put temporary tests, debug functions, or one-off scripts in `src/`, `include/`, `bindings/`, or `python/traditional_dic/`.
- Production code must not depend on `tests/` or `tools/`.
- Do not delete or revert existing user changes unless the user explicitly asks for it.

Windows handoff summary:

- Use Windows x64 with Visual Studio 2022 or Build Tools 2022, MSVC x64, Windows SDK, CMake 3.16+, Git, and Python 3.9+.
- Build from the repository root with CMake.
- Enable Python bindings with `-DTRADITIONAL_DIC_BUILD_PYTHON=ON`.
- Keep generated files out of the source tree.

