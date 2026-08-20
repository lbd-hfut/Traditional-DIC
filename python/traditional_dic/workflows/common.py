"""Small execution-only contracts shared by workflow facade entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..case import ResolvedCase
from ..config_resolver import ResolvedConfig


@dataclass(frozen=True)
class WorkflowContext:
    """Validated inputs and optional output roots for one workflow run."""

    repository_root: Path
    resolved_case: ResolvedCase
    resolved_config: ResolvedConfig
    output_root: Path | None = None
    visualization_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resolved_case, ResolvedCase):
            raise TypeError("resolved_case must be a ResolvedCase")
        if not isinstance(self.resolved_config, ResolvedConfig):
            raise TypeError("resolved_config must be a ResolvedConfig")
        if self.resolved_case.workflow_kind != self.resolved_config.workflow_kind:
            raise ValueError(
                "ResolvedCase and ResolvedConfig workflow kinds differ: "
                f"{self.resolved_case.workflow_kind!r} != {self.resolved_config.workflow_kind!r}"
            )


@dataclass(frozen=True)
class WorkflowRunResult:
    """Workflow output locator plus the normalized F4 contract paths."""

    workflow_kind: str
    output_root: Path
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    run_id: str | None = None
    manifest_path: Path | None = None
    status_path: Path | None = None
    metrics_path: Path | None = None
    result_path: Path | None = None
    contract: Mapping[str, Any] | None = None


def make_context(
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    visualization_root: str | Path | None = None,
) -> WorkflowContext:
    root = Path(repository_root or resolved_case.case_root).resolve()
    return WorkflowContext(
        repository_root=root,
        resolved_case=resolved_case,
        resolved_config=resolved_config,
        output_root=Path(output_root).resolve() if output_root is not None else None,
        visualization_root=Path(visualization_root).resolve() if visualization_root is not None else None,
    )


def require_workflow(context: WorkflowContext, workflow_kind: str) -> None:
    if context.resolved_case.workflow_kind != workflow_kind or context.resolved_config.workflow_kind != workflow_kind:
        raise ValueError(f"workflow facade requires {workflow_kind!r} contracts")


def output_paths(
    context: WorkflowContext,
    solver: str,
    *,
    default_result: str,
    default_visualization: str,
) -> tuple[Path, Path]:
    """Resolve legacy solver roots while honoring an explicit staging root."""
    if context.output_root is not None:
        return (
            context.output_root,
            context.visualization_root or context.output_root / "visualization",
        )
    roots = dict(context.resolved_case.output_roots.get(solver, {}) or {})
    case_root = context.resolved_case.case_root

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else case_root / path

    return (
        rooted(str(roots.get("result_root", default_result))),
        rooted(str(roots.get("visualization_root", default_visualization))),
    )


def execute_with_contract(
    workflow_kind: str,
    resolved_case: ResolvedCase,
    resolved_config: ResolvedConfig,
    *,
    solver_root: str,
    repository_root: str | Path | None,
    output_root: str | Path | None,
    visualization_root: str | Path | None,
    run_id: str | None,
    runner: Any,
) -> WorkflowRunResult:
    """Execute a facade implementation and add the additive F4 metadata layer."""
    from ..run_contract import execute_with_contract as _execute_with_contract

    return _execute_with_contract(
        workflow_kind,
        resolved_case,
        resolved_config,
        solver_root=solver_root,
        repository_root=repository_root,
        output_root=output_root,
        visualization_root=visualization_root,
        run_id=run_id,
        runner=runner,
        make_context_fn=make_context,
        require_workflow_fn=require_workflow,
        output_paths_fn=output_paths,
    )
