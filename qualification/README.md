# Multi-agent qualification

This directory is development/release tooling for S2. It evaluates observable
use of the repository-root `SKILL.md` and the stable `traditional-dic` CLI; it
is not imported by the production package and contains no Agent SDK or
scientific implementation.

The twelve canonical prompts in `scenarios/` are identical for every Agent.
Live Agent sessions are deliberately not part of pytest because providers,
credentials, models, and TTY availability are environmental. Use the harness
to record availability and to run local, deterministic CLI contract probes:

```bash
PYTHONPATH=python \
/home/a306/miniconda3/envs/tradic/bin/python \
qualification/run_qualification.py --probe --json
```

For a release record scoped to one target, add `--agent codex --scenario q01`;
these options only label the record and never invoke a provider. External Agent
sessions are intentionally supplied by the qualification operator.

The harness prefers an installed `traditional-dic` executable and otherwise
uses `PYTHONPATH=python python -m traditional_dic`. External Agent transcripts
must be collected by the release operator; never record credentials or whole
environment dumps. A qualification record is valid only when it includes the
repository HEAD, the `SKILL.md` hash, CLI identity, invocation mode, scenario
evidence, and an explicit availability classification.
