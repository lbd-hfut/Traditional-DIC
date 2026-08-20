# Qualification result records

`s2_qualification_report.json` is a compact, machine-readable release record
containing repository/Skill/CLI identities, scenario metadata, executable
availability, and local CLI probe evidence. `agent_session_evidence.json`
records the provider/TTY limitations observed during this qualification. Raw
transcripts are intentionally not committed and must never contain credentials.

An Agent is not marked qualified by executable discovery alone. A release
operator may attach a fresh, secret-redacted transcript and replace the
corresponding `BLOCKED_ENVIRONMENT` scenario states after an actual session.
