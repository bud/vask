"""Workflow engine — multi-step automation pipelines."""

from __future__ import annotations

import ast
import operator
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vask.logging import Span, audit, get_logger

logger = get_logger("workflows")


# ── Safe expression evaluator (no eval/exec) ──────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_FUNCTIONS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "upper": lambda s: s.upper() if isinstance(s, str) else str(s).upper(),
    "lower": lambda s: s.lower() if isinstance(s, str) else str(s).lower(),
    "strip": lambda s: s.strip() if isinstance(s, str) else str(s).strip(),
}


def safe_eval(expression: str, variables: dict[str, Any]) -> Any:
    """Evaluate a simple expression safely using AST parsing.

    Supports: variable references, string/number literals, comparisons,
    boolean ops, 'in'/'not in', function calls (len, str, int, etc.),
    attribute access on strings (.upper(), .lower(), .strip(), .split()).
    Does NOT support: arbitrary code execution, imports, lambdas, comprehensions.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        op_func = _SAFE_OPS.get(type(node.op))
        if not op_func:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)
    elif isinstance(node, ast.BoolOp):
        values = [_eval_node(v, variables) for v in node.values]
        op_func = _SAFE_OPS.get(type(node.op))
        if not op_func:
            raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")
        result = values[0]
        for v in values[1:]:
            result = op_func(result, v)
        return result
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.Not):
            return not operand
        elif isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
    elif isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            op_func = _SAFE_OPS.get(type(op))
            if not op_func:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            if not op_func(left, right):
                return False
            left = right
        return True
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func = _SAFE_FUNCTIONS.get(node.func.id)
            if not func:
                raise ValueError(f"Unsupported function: {node.func.id}")
            args = [_eval_node(a, variables) for a in node.args]
            return func(*args)
        elif isinstance(node.func, ast.Attribute):
            obj = _eval_node(node.func.value, variables)
            method_name = node.func.attr
            _allowed = ("upper", "lower", "strip", "split",
                       "startswith", "endswith", "replace", "count")
            if isinstance(obj, str) and method_name in _allowed:
                method = getattr(obj, method_name)
                args = [_eval_node(a, variables) for a in node.args]
                return method(*args)
            raise ValueError(f"Unsupported method: {method_name}")
        raise ValueError(f"Unsupported call: {ast.dump(node.func)}")
    elif isinstance(node, ast.Attribute):
        obj = _eval_node(node.value, variables)
        if isinstance(obj, dict):
            return obj.get(node.attr, "")
        raise ValueError(f"Attribute access only supported on dicts, got {type(obj).__name__}")
    elif isinstance(node, ast.Subscript):
        obj = _eval_node(node.value, variables)
        if isinstance(node.slice, ast.Constant):
            return obj[node.slice.value]
        idx = _eval_node(node.slice, variables)
        return obj[idx]
    elif isinstance(node, ast.List):
        return [_eval_node(e, variables) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, variables) for e in node.elts)
    elif isinstance(node, ast.IfExp):
        test = _eval_node(node.test, variables)
        if test:
            return _eval_node(node.body, variables)
        return _eval_node(node.orelse, variables)
    elif isinstance(node, ast.JoinedStr):
        # f-string support
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append(str(_eval_node(v.value, variables)))
            else:
                parts.append(str(_eval_node(v, variables)))
        return "".join(parts)
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── Data types ─────────────────────────────────────────────────────────

class StepType(StrEnum):
    """Types of workflow steps."""

    ASK = "ask"  # Send text to LLM
    TOOL = "tool"  # Execute a tool
    TRANSFORM = "transform"  # Transform data with a safe expression
    CONDITION = "condition"  # Branch based on condition
    PARALLEL = "parallel"  # Run steps in parallel
    HUMAN_APPROVAL = "human_approval"  # Wait for human confirmation
    WEBHOOK_NOTIFY = "webhook_notify"  # Send a webhook/notification


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


class WorkflowStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"  # Waiting for human approval


@dataclass(slots=True)
class StepDef:
    """Definition of a single workflow step."""

    id: str
    name: str
    type: StepType
    config: dict[str, Any] = field(default_factory=dict)
    # Config varies by type:
    #   ask: {"text": "...", "llm": "claude", "system_prompt": "..."}
    #   tool: {"tool_name": "slack_send_message", "params": {...}}
    #   transform: {"expression": "result.upper()", "input_from": "step_1"}
    #   condition: {"expression": "'error' not in result",
    #              "if_true": "step_3", "if_false": "step_4"}
    #   parallel: {"steps": ["step_a", "step_b"]}
    #   human_approval: {"message": "Approve sending email to 500 users?"}
    #   webhook_notify: {"url": "...", "method": "POST"}
    depends_on: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 1
    timeout: float = 60.0


@dataclass(slots=True)
class StepResult:
    """Result of executing a workflow step."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowDef:
    """Definition of a complete workflow."""

    id: str
    name: str
    description: str = ""
    steps: list[StepDef] = field(default_factory=list)
    trigger: str = ""  # "webhook:<id>", "schedule:cron", "manual"
    variables: dict[str, Any] = field(default_factory=dict)  # Default variables
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowRun:
    """A single execution of a workflow."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.RUNNING
    variables: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, StepResult] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error: str = ""
    current_step: str = ""


class WorkflowEngine:
    """Executes multi-step workflows with tool integration."""

    def __init__(self, pipeline: Any, tool_registry: Any) -> None:
        self._pipeline = pipeline
        self._tool_registry = tool_registry
        self._workflows: dict[str, WorkflowDef] = {}
        self._runs: dict[str, WorkflowRun] = {}
        self._approval_callbacks: dict[str, bool | None] = {}

    def register_workflow(self, workflow: WorkflowDef) -> None:
        """Register a workflow definition."""
        self._workflows[workflow.id] = workflow
        logger.info(f"Registered workflow '{workflow.name}' (id={workflow.id})")

    def register_from_dict(self, data: dict[str, Any]) -> WorkflowDef:
        """Register a workflow from a dictionary (e.g., YAML/JSON parsed)."""
        steps = []
        for s in data.get("steps", []):
            steps.append(StepDef(
                id=s["id"],
                name=s.get("name", s["id"]),
                type=StepType(s["type"]),
                config=s.get("config", {}),
                depends_on=s.get("depends_on", []),
                max_retries=s.get("max_retries", 1),
                timeout=s.get("timeout", 60.0),
            ))

        workflow = WorkflowDef(
            id=data.get("id", uuid.uuid4().hex[:12]),
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            trigger=data.get("trigger", "manual"),
            variables=data.get("variables", {}),
        )
        self.register_workflow(workflow)
        return workflow

    async def execute(
        self,
        workflow_id: str,
        variables: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """Execute a workflow."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        run = WorkflowRun(
            workflow_id=workflow_id,
            variables={**workflow.variables, **(variables or {})},
        )
        self._runs[run.id] = run

        audit.log("workflow", workflow.name, {"run_id": run.id, "workflow_id": workflow_id})
        logger.info(f"Starting workflow '{workflow.name}' (run={run.id})")

        with Span("workflow_execute") as span:
            span.attributes["workflow_id"] = workflow_id
            span.attributes["run_id"] = run.id

            try:
                execution_order = self._resolve_order(workflow.steps)

                for step_def in execution_order:
                    if run.status in (WorkflowStatus.PAUSED, WorkflowStatus.FAILED):
                        break

                    # Check if dependencies completed
                    deps_ok = all(
                        run.step_results.get(dep, StepResult(step_id=dep)).status
                        == StepStatus.COMPLETED
                        for dep in step_def.depends_on
                    )
                    if not deps_ok:
                        result = StepResult(step_id=step_def.id, status=StepStatus.SKIPPED)
                        run.step_results[step_def.id] = result
                        continue

                    run.current_step = step_def.id
                    result = await self._execute_step(step_def, run)
                    run.step_results[step_def.id] = result

                    # Store step output as a variable for downstream steps
                    run.variables[f"step_{step_def.id}"] = result.output

                    if result.status == StepStatus.FAILED:
                        run.status = WorkflowStatus.FAILED
                        run.error = f"Step '{step_def.name}' failed: {result.error}"
                    elif result.status == StepStatus.WAITING_APPROVAL:
                        run.status = WorkflowStatus.PAUSED

                if run.status == WorkflowStatus.RUNNING:
                    run.status = WorkflowStatus.COMPLETED

            except Exception as e:
                run.status = WorkflowStatus.FAILED
                run.error = str(e)
                logger.error(f"Workflow '{workflow.name}' failed: {e}", exc_info=True)

            run.completed_at = time.time()
            span.attributes["status"] = run.status.value

        return run

    async def _execute_step(
        self,
        step_def: StepDef,
        run: WorkflowRun,
    ) -> StepResult:
        """Execute a single workflow step."""
        result = StepResult(
            step_id=step_def.id,
            status=StepStatus.RUNNING,
            started_at=time.time(),
        )

        logger.info(f"Executing step '{step_def.name}' (type={step_def.type.value})")

        try:
            if step_def.type == StepType.ASK:
                output = await self._step_ask(step_def, run)
            elif step_def.type == StepType.TOOL:
                output = await self._step_tool(step_def, run)
            elif step_def.type == StepType.TRANSFORM:
                output = self._step_transform(step_def, run)
            elif step_def.type == StepType.CONDITION:
                output = self._step_condition(step_def, run)
            elif step_def.type == StepType.HUMAN_APPROVAL:
                output = await self._step_human_approval(step_def, run)
                if output == "__WAITING__":
                    result.status = StepStatus.WAITING_APPROVAL
                    result.completed_at = time.time()
                    return result
            elif step_def.type == StepType.WEBHOOK_NOTIFY:
                output = await self._step_webhook_notify(step_def, run)
            else:
                output = f"Unknown step type: {step_def.type}"

            result.output = output
            result.status = StepStatus.COMPLETED

        except Exception as e:
            result.error = str(e)
            result.status = StepStatus.FAILED
            logger.error(f"Step '{step_def.name}' failed: {e}")

        result.completed_at = time.time()
        result.duration_ms = round(
            (result.completed_at - result.started_at) * 1000, 2
        )
        return result

    async def _step_ask(self, step: StepDef, run: WorkflowRun) -> str:
        """Execute an LLM ask step."""
        text = self._interpolate(step.config.get("text", ""), run.variables)
        llm = step.config.get("llm")
        system_prompt = step.config.get("system_prompt")

        response = await self._pipeline.ask(
            text,
            llm_name=llm,
            output_name="json",
            system_prompt=system_prompt,
        )
        return response.text

    async def _step_tool(self, step: StepDef, run: WorkflowRun) -> str:
        """Execute a tool step."""
        tool_name = step.config.get("tool_name", "")
        params = step.config.get("params", {})

        # Interpolate variables into params
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved_params[k] = self._interpolate(v, run.variables)
            else:
                resolved_params[k] = v

        result = await self._tool_registry.execute(
            tool_name, resolved_params, tool_call_id=f"wf_{run.id}_{step.id}"
        )
        if result.is_error:
            raise RuntimeError(f"Tool error: {result.output}")
        return result.output

    def _step_transform(self, step: StepDef, run: WorkflowRun) -> str:
        """Transform data using AST-based safe expression evaluation."""
        input_from = step.config.get("input_from", "")
        input_data = run.variables.get(f"step_{input_from}", "")
        expression = step.config.get("expression", "result")

        eval_vars = {
            "result": input_data,
            **{k: v for k, v in run.variables.items()},
        }
        return str(safe_eval(expression, eval_vars))

    def _step_condition(self, step: StepDef, run: WorkflowRun) -> str:
        """Evaluate a condition and return the branch to take."""
        expression = step.config.get("expression", "True")
        input_from = step.config.get("input_from", "")
        input_data = run.variables.get(f"step_{input_from}", "")

        eval_vars = {
            "result": input_data,
            **{k: v for k, v in run.variables.items()},
        }
        condition = bool(safe_eval(expression, eval_vars))

        if condition:
            return step.config.get("if_true", "continue")
        else:
            return step.config.get("if_false", "skip")

    async def _step_human_approval(self, step: StepDef, run: WorkflowRun) -> str:
        """Request human approval (returns __WAITING__ if not yet approved)."""
        approval_key = f"{run.id}:{step.id}"
        approval = self._approval_callbacks.get(approval_key)

        if approval is None:
            self._approval_callbacks[approval_key] = None
            logger.info(f"Waiting for human approval: {step.config.get('message', '')}")
            return "__WAITING__"
        elif approval:
            return "approved"
        else:
            raise RuntimeError("Human rejected the step")

    async def _step_webhook_notify(self, step: StepDef, run: WorkflowRun) -> str:
        """Send an outbound webhook notification."""
        import httpx

        url = self._interpolate(step.config.get("url", ""), run.variables)
        method = step.config.get("method", "POST")
        body = {
            "workflow_id": run.workflow_id,
            "run_id": run.id,
            "step_id": step.id,
            "variables": {k: str(v)[:500] for k, v in run.variables.items()},
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url, json=body)
            return f"Status: {resp.status_code}"

    def approve_step(self, run_id: str, step_id: str, approved: bool = True) -> bool:
        """Approve or reject a pending human approval step."""
        key = f"{run_id}:{step_id}"
        if key in self._approval_callbacks:
            self._approval_callbacks[key] = approved
            return True
        return False

    def _interpolate(self, template: str, variables: dict[str, Any]) -> str:
        """Simple variable interpolation: {{var_name}} -> value."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _resolve_order(self, steps: list[StepDef]) -> list[StepDef]:
        """Topological sort of steps by dependencies."""
        step_map = {s.id: s for s in steps}
        visited: set[str] = set()
        order: list[StepDef] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if not step:
                return
            for dep in step.depends_on:
                visit(dep)
            order.append(step)

        for s in steps:
            visit(s.id)
        return order

    # ── Run management ─────────────────────────────────────────────────

    def get_run(self, run_id: str) -> WorkflowRun | None:
        return self._runs.get(run_id)

    def list_runs(self, workflow_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        runs = list(self._runs.values())
        if workflow_id:
            runs = [r for r in runs if r.workflow_id == workflow_id]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return [
            {
                "id": r.id,
                "workflow_id": r.workflow_id,
                "status": r.status.value,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "current_step": r.current_step,
                "error": r.error,
                "steps_completed": sum(
                    1 for s in r.step_results.values()
                    if s.status == StepStatus.COMPLETED
                ),
                "steps_total": len(r.step_results),
            }
            for r in runs[:limit]
        ]

    def list_workflows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "trigger": w.trigger,
                "steps_count": len(w.steps),
                "created_at": w.created_at,
            }
            for w in self._workflows.values()
        ]

    def get_workflow(self, workflow_id: str) -> WorkflowDef | None:
        return self._workflows.get(workflow_id)
