"""Pre-submission preflight (job "dry run" / health check).

A job used to start running the instant it was POSTed, so an illegal parameter
or an empty/shape-mismatched dataset was only discovered mid-run.  The
preflight performs — **without scheduling or executing any real computation** —
as complete a health check as possible:

1. ``params``      — every submitted field parses, has a legal value and
                     satisfies the mapper/reducer's declared parameter contract;
2. ``data``        — the input dataset that *would* be generated exists, is
                     non-empty and every planned shard carries records;
3. ``shape``       — the generated record shape matches the mapper's declared
                     input, and a **canary trial run** over the first shards
                     actually invokes the mapper and reducer on real records
                     (no files written, no tasks created) to catch crashes,
                     unhashable keys or value-shape mismatches directly at the
                     offending field/shard;
4. ``sharding``    — the requested map/reduce split is feasible for this
                     dataset (clamping, empty shards, over-partitioning);
5. ``dependencies``— every declared upstream job exists and is in a state that
                     satisfies the start condition, and no dependency cycle is
                     introduced;
6. ``cluster``     — at least one live worker could take the work;
7. ``storage``     — the job's shard directory is writable.

Every finding is one :class:`Check` with a severity, a machine-checked group, a
human bilingual message and a precise ``field`` / ``target`` locator (form
field name, shard id, parameter name or upstream job id) so the UI can point
the user at the exact thing to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.common import constants as C
from backend.common.hashing import partition_for
from backend.common.models import Job, new_job
from backend.common.storage import Storage
from backend.master.shard_planner import ShardPlanner
from backend.tasks import registry as reg
from backend.tasks.samples import input_kind_for, input_shape_for

ERROR = "error"       # blocks submission
WARNING = "warning"   # allowed to submit, but likely wrong
INFO = "info"         # informational notice

# Submission-wide hard bounds (mirror the HTML inputs; kept server-side so a
# direct API caller is validated identically).
BOUNDS = {
    "num_map_tasks": (1, 1000),
    "num_reduce_tasks": (1, 500),
    "input_rows": (1, 10_000_000),
}

# Reducer value shapes that are compatible with what a mapper emits.  A missing
# edge is only a warning: built-in reducers coerce values, and custom pairings
# (e.g. distinct + count) are intentionally legitimate.
EMIT_ACCEPTS: dict[str, set[str]] = {
    "count": {"count", "number", "any"},
    "number": {"number", "any"},
    "any": {"count", "number", "any"},
}

# How many of the first shards / records the canary actually exercises. The
# point is a cheap representative probe, not the real computation.
CANARY_SHARDS = 2
CANARY_RECORDS_PER_SHARD = 64


@dataclass
class Finding:
    severity: str
    group: str
    message: str
    field: str = ""          # form/JSON field the problem belongs to
    target: str = ""         # concrete locator: shard id, param name, job id…
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity, "group": self.group,
            "message": self.message, "field": self.field,
            "target": self.target, "suggestion": self.suggestion,
        }


@dataclass
class PreflightResult:
    ok: bool
    findings: list[Finding] = field(default_factory=list)
    # The plan that *would* be submitted (counts and shard sizes) — useful
    # evidence for the user even when everything passes.
    plan_preview: dict = field(default_factory=dict)
    normalized: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    def add(self, severity: str, group: str, message: str, **kw: Any) -> None:
        self.findings.append(Finding(severity, group, message, **kw))

    def to_dict(self) -> dict:
        groups: dict[str, list[dict]] = {}
        for f in self.findings:
            groups.setdefault(f.group, []).append(f.to_dict())
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
            "groups": groups,
            "plan_preview": self.plan_preview,
            "normalized": self.normalized,
        }


# ---------------------------------------------------------------------------
# Payload normalization (shared with the real submission path)
# ---------------------------------------------------------------------------
def _parse_int(value: Any, field_name: str, result: PreflightResult,
               default: int) -> Optional[int]:
    """Strict integer parse: reject NaN/blank/non-numeric strings on the field."""
    if value is None or (isinstance(value, str) and not value.strip()):
        result.add(ERROR, "params", f"{field_name} 不能为空 / is required",
                   field=field_name, suggestion=f"使用默认值 {default}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        result.add(ERROR, "params",
                   f"{field_name} 必须是整数 / must be an integer, got {type(value).__name__}",
                   field=field_name)
        return None
    try:
        iv = int(str(value).strip(), 10)
    except (TypeError, ValueError):
        result.add(ERROR, "params",
                   f"{field_name} 不是合法整数 / not a valid integer: {value!r}",
                   field=field_name)
        return None
    lo, hi = BOUNDS[field_name]
    if not lo <= iv <= hi:
        result.add(ERROR, "params",
                   f"{field_name}={iv} 超出允许范围 [{lo}, {hi}] / out of range",
                   field=field_name, suggestion=f"取 {lo}~{hi} 之间的整数")
        return None
    return iv


def normalize_payload(payload: dict, defaults: dict,
                      result: PreflightResult) -> Optional[dict]:
    """Validate and normalize the raw submission body."""
    defaults = defaults or {}
    name = str(payload.get("name") or "untitled").strip() or "untitled"

    mapper = str(payload.get("mapper") or "")
    if not mapper:
        result.add(ERROR, "params", "未选择 Mapper / mapper is required", field="mapper")
    elif not reg.has_mapper(mapper):
        result.add(ERROR, "params", f"未知的 mapper: {mapper!r} / unknown mapper",
                   field="mapper", target=mapper,
                   suggestion=f"可选: {', '.join(s['name'] for s in reg.list_mappers())}")

    reducer = str(payload.get("reducer") or "")
    if not reducer:
        result.add(ERROR, "params", "未选择 Reducer / reducer is required", field="reducer")
    elif not reg.has_reducer(reducer):
        result.add(ERROR, "params", f"未知的 reducer: {reducer!r} / unknown reducer",
                   field="reducer", target=reducer,
                   suggestion=f"可选: {', '.join(s['name'] for s in reg.list_reducers())}")

    num_map = _parse_int(payload.get("num_map_tasks", defaults.get("num_map_tasks", 8)),
                         "num_map_tasks", result, 8)
    num_reduce = _parse_int(payload.get("num_reduce_tasks", defaults.get("num_reduce_tasks", 4)),
                            "num_reduce_tasks", result, 4)
    input_rows = _parse_int(payload.get("input_rows", defaults.get("input_rows", 12000)),
                            "input_rows", result, 12000)

    raw_params = payload.get("params")
    params: dict = {}
    if raw_params is not None and not isinstance(raw_params, dict):
        result.add(ERROR, "params", "params 必须是对象 / must be a JSON object",
                   field="params")
    else:
        params = dict(raw_params or {})

    depends_on = _normalize_depends(payload, result)

    if result.errors:
        return None

    normalized = {
        "name": name,
        "mapper": mapper,
        "reducer": reducer,
        "num_map_tasks": num_map,
        "num_reduce_tasks": num_reduce,
        "input_rows": input_rows,
        "params": params,
        "depends_on": depends_on,
    }
    result.normalized = normalized
    return normalized


def _normalize_depends(payload: dict, result: PreflightResult) -> list[str]:
    raw = payload.get("depends_on") or []
    if isinstance(raw, str):
        raw = [raw] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        result.add(ERROR, "dependencies",
                   "depends_on 必须是作业 ID 列表 / must be a list of job ids",
                   field="depends_on")
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            result.add(ERROR, "dependencies",
                       f"非法的前置作业 ID / invalid dependency entry: {item!r}",
                       field="depends_on", target=str(item))
            continue
        item = item.strip()
        if item in out:
            result.add(INFO, "dependencies",
                       f"前置作业 {item} 重复声明，已去重 / duplicate dependency removed",
                       field="depends_on", target=item)
            continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# The preflight itself
# ---------------------------------------------------------------------------
class Preflight:
    def __init__(self, storage: Storage, planner: ShardPlanner, job_manager,
                 registry, config) -> None:
        self.storage = storage
        self.planner = planner
        self.job_manager = job_manager
        self.registry = registry
        self.config = config

    def run(self, payload: dict, defaults: dict | None = None) -> PreflightResult:
        result = PreflightResult(ok=False)

        # 1. Field/parameter validation (everything else depends on it).
        normalized = normalize_payload(payload, defaults or {}, result)
        if normalized is None:
            # Still try cluster/dependency checks that don't need the plan.
            self._check_cluster(result)
            self._check_dependencies(payload, candidate_id="", result=result,
                                     planned={})
            self._finalize(result)
            return result

        self._check_param_contract(normalized, result)

        # Build a throwaway Job purely to drive the deterministic planner;
        # nothing is persisted.
        probe = new_job(
            normalized["name"], normalized["mapper"], normalized["reducer"],
            normalized["num_map_tasks"], normalized["num_reduce_tasks"],
            normalized["input_rows"], dict(normalized["params"]),
        )
        probe.params["input_kind"] = input_kind_for(normalized["mapper"])

        try:
            dry = self.planner.dry_plan(probe)
        except Exception as exc:  # noqa: BLE001 - a planner crash is a finding, not a 500
            result.add(ERROR, "data",
                       f"输入数据生成失败 / input generation failed: {type(exc).__name__}: {exc}",
                       field="input_rows")
            self._check_cluster(result)
            self._finalize(result)
            return result

        result.plan_preview = {
            "input_kind": dry["kind"],
            "input_rows": len(dry["records"]),
            "num_map_tasks": len(dry["chunks"]),
            "num_reduce_tasks": normalized["num_reduce_tasks"],
            "shard_sizes": [len(c) for c in dry["chunks"]],
            "empty_shards": [dry["input_shards"][i]
                             for i, c in enumerate(dry["chunks"]) if not c],
        }

        # 2–4. Data presence/non-emptiness, shape match + canary, sharding.
        self._check_data(normalized, dry, result)
        self._check_shape_and_canary(normalized, dry, result)
        self._check_sharding(normalized, dry, result)

        # 5–7. Dependencies, cluster readiness, storage writability.
        self._check_dependencies(normalized, candidate_id=probe.job_id,
                                 result=result, planned=dry)
        self._check_cluster(result)
        self._check_storage(probe, result)

        self._finalize(result)
        return result

    # ------------------------------------------------------------------
    # 1b. Declared parameter contract
    # ------------------------------------------------------------------
    def _check_param_contract(self, n: dict, result: PreflightResult) -> None:
        mspec = reg.mapper_spec(n["mapper"])
        rspec = reg.reducer_spec(n["reducer"])
        declared = {p.name: p for p in (mspec.params if mspec else [])}
        declared.update({p.name: p for p in (rspec.params if rspec else [])})
        params = n["params"]

        # Internal/reserved params are never user-facing.
        reserved = {"input_kind", "simulate_failure"}

        for pname, pspec in declared.items():
            if pspec.required and pname not in params:
                result.add(ERROR, "params",
                           f"缺少必填参数 params.{pname} / missing required parameter "
                           f"({pspec.description})",
                           field=f"params.{pname}", target=pname,
                           suggestion="在高级参数中填写")
            elif pname in params:
                self._check_param_type(pspec, params[pname], result)

        for key in params:
            if key in reserved or key in declared:
                continue
            result.add(INFO, "params",
                       f"参数 params.{key} 未被 mapper/reducer 使用，将被忽略 / "
                       f"unrecognized parameter ignored",
                       field="params", target=key)

        if params.get("simulate_failure"):
            result.add(INFO, "params",
                       "已开启故障注入：每个任务的首次尝试将失败并自动重试 / "
                       "fault injection enabled — first attempt of every task will fail",
                       field="params.simulate_failure")

    def _check_param_type(self, pspec: reg.ParamSpec, value: Any,
                          result: PreflightResult) -> None:
        field_name = f"params.{pspec.name}"
        if pspec.type == "number":
            try:
                float(value)
            except (TypeError, ValueError):
                result.add(ERROR, "params",
                           f"params.{pspec.name} 必须是数字 / must be numeric, got {value!r}",
                           field=field_name, target=pspec.name)
        elif pspec.type == "bool":
            if not isinstance(value, bool):
                result.add(WARNING, "params",
                           f"params.{pspec.name} 应为布尔值 / should be boolean",
                           field=field_name, target=pspec.name)
        elif pspec.type == "string":
            if not isinstance(value, str) or not value.strip():
                result.add(ERROR, "params",
                           f"params.{pspec.name} 必须是非空字符串 / must be a non-empty string",
                           field=field_name, target=pspec.name)

    # ------------------------------------------------------------------
    # 2. Data existence / non-emptiness
    # ------------------------------------------------------------------
    def _check_data(self, n: dict, dry: dict, result: PreflightResult) -> None:
        records = dry["records"]
        if not records:
            result.add(ERROR, "data",
                       "输入数据为空：生成器没有产生任何记录 / input dataset is empty",
                       field="input_rows", target="input",
                       suggestion="增大 input_rows 或更换 mapper")
            return

        result.add(INFO, "data",
                   f"输入数据集就绪：{len(records)} 条记录，将写入 {len(dry['chunks'])} 个输入分片 / "
                   f"dataset ready: {len(records)} records across {len(dry['chunks'])} shards",
                   target="input")

        empties = [dry["input_shards"][i]
                   for i, chunk in enumerate(dry["chunks"]) if not chunk]
        if empties:
            result.add(ERROR, "data",
                       f"{len(empties)} 个输入分片为空，对应 map 任务将没有任何输入 / "
                       f"{len(empties)} input shard(s) are empty",
                       field="num_map_tasks", target=", ".join(empties[:5]),
                       suggestion="减少 num_map_tasks 或增大 input_rows")

    # ------------------------------------------------------------------
    # 3. Shape match + canary trial run
    # ------------------------------------------------------------------
    def _check_shape_and_canary(self, n: dict, dry: dict,
                                result: PreflightResult) -> None:
        kind = dry["kind"]
        records = dry["records"]
        expected_shape = input_shape_for(kind)
        mapper = reg.get_mapper(n["mapper"])
        reducer = reg.get_reducer(n["reducer"])
        mspec = reg.mapper_spec(n["mapper"])
        rspec = reg.reducer_spec(n["reducer"])

        # 3a. Declared dataset shape vs. mapper contract.
        if mspec and mspec.input_shape not in ("any", expected_shape):
            result.add(ERROR, "shape",
                       f"数据形状不匹配：{n['mapper']} 需要 {mspec.input_shape} 输入，"
                       f"但本作业生成的是 {expected_shape} 数据 / "
                       f"mapper expects {mspec.input_shape} records but dataset is {expected_shape}",
                       field="mapper", target="input")

        # 3b. Declared mapper output vs. reducer accepted value shape.
        if mspec and rspec:
            compatible = EMIT_ACCEPTS.get(mspec.emits, set())
            if rspec.accepts not in compatible:
                result.add(WARNING, "shape",
                           f"输出形状可能不匹配：mapper 产出 {mspec.emits} 值，"
                           f"reducer {n['reducer']} 期望 {rspec.accepts} 值 / "
                           f"mapper emits {mspec.emits} but reducer accepts {rspec.accepts}",
                           field="reducer",
                           suggestion="检查 mapper/reducer 搭配是否为预期组合")

        # 3c. Canary: actually run the mapper on a small representative probe of
        # the real generated records — no shard files, no tasks, no scheduling.
        params = dict(n["params"])
        params["input_kind"] = kind
        probe_pairs: list[tuple[Any, Any]] = []
        sampled_shards = dry["chunks"][:CANARY_SHARDS]
        sampled = 0
        for i, chunk in enumerate(sampled_shards):
            shard_id = dry["input_shards"][i]
            probe_records = chunk[:CANARY_RECORDS_PER_SHARD]
            try:
                pairs = mapper(probe_records, params)
            except Exception as exc:  # noqa: BLE001 - surface the exact crash
                result.add(ERROR, "shape",
                           f"试运行 mapper 在分片 {shard_id} 的前 {len(probe_records)} 条记录上抛异常："
                           f"{type(exc).__name__}: {exc} / mapper canary raised",
                           field="mapper", target=shard_id)
                return
            if not isinstance(pairs, list):
                result.add(ERROR, "shape",
                           f"mapper 必须返回 (key, value) 列表，实际返回 {type(pairs).__name__} / "
                           f"mapper must return a list of pairs",
                           field="mapper", target=shard_id)
                return
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    result.add(ERROR, "shape",
                               f"mapper 在分片 {shard_id} 产出非法条目（应为 [key, value]）：{pair!r} / "
                               f"mapper emitted a malformed pair",
                               field="mapper", target=shard_id)
                    continue
                key, value = pair
                try:
                    hash(key)
                except TypeError:
                    result.add(ERROR, "shape",
                               f"mapper 在分片 {shard_id} 产出不可哈希的 key（无法分区/分组）：{key!r} / "
                               f"unhashable key cannot be partitioned",
                               field="mapper", target=shard_id)
                    continue
                probe_pairs.append((key, value))
                # The partitioner is the exact one used at runtime.
                partition_for(key, n["num_reduce_tasks"])
            sampled += len(probe_records)

        if sampled == 0:
            return  # empty-shard error already reported by the data group

        if not probe_pairs:
            verb = "匹配不到任何记录，作业将成功但结果为空" if n["mapper"] == "grep_mapper" \
                else "对探测数据没有任何输出，结果将为空"
            result.add(WARNING, "shape",
                       f"试运行 mapper 处理 {sampled} 条真实记录后未产出任何键值对：{verb} / "
                       f"canary mapper produced zero pairs — the result will be empty",
                       field="params.pattern" if n["mapper"] == "grep_mapper" else "mapper",
                       suggestion="grep 作业请检查 params.pattern 是否拼错")
            return

        # 3d. Canary the reducer the same way the shuffle/group stage would:
        # group probe pairs by key in sorted order and invoke the reducer on the
        # first few groups, so a crashing reducer is caught before submission.
        grouped: dict[Any, list[Any]] = {}
        for key, value in probe_pairs:
            grouped.setdefault(key, []).append(value)

        groups_run = 0
        for key in sorted(grouped, key=lambda k: str(k)):
            try:
                out = reducer(key, grouped[key], params)
            except Exception as exc:  # noqa: BLE001
                result.add(ERROR, "shape",
                           f"试运行 reducer 在 key={key!r}（{len(grouped[key])} 个值）上抛异常："
                           f"{type(exc).__name__}: {exc} / reducer canary raised",
                           field="reducer", target=f"key={key}")
                return
            if not isinstance(out, dict) or "key" not in out:
                result.add(ERROR, "shape",
                           f"reducer 必须返回包含 'key' 的 dict，key={key!r} 实际返回 {out!r} / "
                           f"reducer must return a dict containing 'key'",
                           field="reducer", target=f"key={key}")
                return
            groups_run += 1
            if groups_run >= 8:
                break

        result.add(INFO, "shape",
                   f"试运行通过：mapper 已处理 {sampled} 条真实记录、产出 {len(probe_pairs)} 个键值对，"
                   f"reducer 已验证 {groups_run} 个分组（未执行真正计算、未落盘）/ "
                   f"canary passed on {sampled} records / {len(probe_pairs)} pairs / "
                   f"{groups_run} groups (no real computation, nothing persisted)",
                   target="canary")

    # ------------------------------------------------------------------
    # 4. Sharding feasibility
    # ------------------------------------------------------------------
    def _check_sharding(self, n: dict, dry: dict, result: PreflightResult) -> None:
        requested_map = n["num_map_tasks"]
        actual_map = len(dry["chunks"])
        rows = len(dry["records"])

        if requested_map > rows:
            result.add(WARNING, "sharding",
                       f"切分不可细于数据：请求 {requested_map} 个 map 任务但只有 {rows} 条记录，"
                       f"将自动收敛为 {actual_map} 个 / map tasks clamped to record count",
                       field="num_map_tasks",
                       suggestion=f"将 num_map_tasks 调到 ≤ {rows}")

        sizes = [len(c) for c in dry["chunks"]]
        if sizes and max(sizes) - min(sizes) > 1:
            result.add(WARNING, "sharding",
                       f"分片负载不均：最大 {max(sizes)} 条 / 最小 {min(sizes)} 条 / "
                       f"uneven shard sizes",
                       target="split")

        num_reduce = n["num_reduce_tasks"]
        if num_reduce > max(1, rows // 10):
            result.add(WARNING, "sharding",
                       f"reduce 任务数（{num_reduce}）相对输入量偏大，多数分区可能为空 / "
                       f"many reduce partitions may be empty for this input size",
                       field="num_reduce_tasks",
                       suggestion="减少 num_reduce_tasks")

        result.add(INFO, "sharding",
                   f"切分规划可行：{actual_map} map × {num_reduce} reduce，"
                   f"分片大小 {min(sizes)}~{max(sizes)} 条 / split feasible",
                   target="split")

    # ------------------------------------------------------------------
    # 5. Upstream dependency readiness
    # ------------------------------------------------------------------
    def _check_dependencies(self, payload: dict, candidate_id: str,
                            result: PreflightResult, planned: dict) -> None:
        deps = payload.get("depends_on") if isinstance(payload, dict) else None
        if not deps:
            return
        deps = [d for d in deps if isinstance(d, str) and d.strip()]
        if not deps:
            return

        statuses: dict[str, Job] = {}
        for dep in deps:
            upstream = self.job_manager.get_job(dep)
            if upstream is None:
                result.add(ERROR, "dependencies",
                           f"前置作业 {dep} 不存在 / upstream job does not exist",
                           field="depends_on", target=dep,
                           suggestion="检查作业 ID，或先提交该前置作业")
                continue
            statuses[dep] = upstream
            if upstream.status == C.JOB_SUCCEEDED:
                result.add(INFO, "dependencies",
                           f"前置作业 {dep}（{upstream.name}）已成功，启动条件满足 / "
                           f"upstream {dep} succeeded",
                           field="depends_on", target=dep)
            elif upstream.status in (C.JOB_FAILED, C.JOB_CANCELLED):
                result.add(ERROR, "dependencies",
                           f"前置作业 {dep}（{upstream.name}）状态为 {upstream.status}，"
                           f"不具备启动条件 / upstream is {upstream.status}",
                           field="depends_on", target=dep,
                           suggestion="重新运行该前置作业或解除依赖")
            else:
                result.add(WARNING, "dependencies",
                           f"前置作业 {dep}（{upstream.name}）仍在运行（{upstream.status}），"
                           f"提交后作业将等待其成功 / upstream still running; job will wait",
                           field="depends_on", target=dep)

        # Cycle detection over the existing DAG plus this candidate's edges.
        graph: dict[str, list[str]] = {}
        for job in self.job_manager.list_jobs():
            graph[job.job_id] = list(getattr(job, "depends_on", []) or [])
        if candidate_id:
            graph[candidate_id] = deps
        cycle = _find_cycle_touching(graph, set(statuses) | {candidate_id})
        if cycle:
            chain = " -> ".join(cycle)
            result.add(ERROR, "dependencies",
                       f"依赖关系存在环：{chain} / dependency cycle detected",
                       field="depends_on", target=chain)

    # ------------------------------------------------------------------
    # 6. Cluster readiness
    # ------------------------------------------------------------------
    def _check_cluster(self, result: PreflightResult) -> None:
        alive = self.registry.alive()
        all_workers = self.registry.all()
        if not alive:
            # Not a hard error: the job can queue in PENDING/MAP and the
            # scheduler dispatches once a worker registers. Still loud, because
            # "submitted but nothing happens" is exactly the surprise to avoid.
            result.add(WARNING, "cluster",
                       "当前没有存活的 Worker，作业会先排队，待 Worker 注册后才开始执行 / "
                       "no alive workers — the job will queue until one registers",
                       field="_cluster",
                       suggestion="启动至少一个 Worker：python -m backend.app worker ...")
            return
        result.add(INFO, "cluster",
                   f"集群就绪：{len(alive)} 个存活 Worker / {len(alive)} alive worker(s)",
                   target="_cluster")

    # ------------------------------------------------------------------
    # 7. Storage writability
    # ------------------------------------------------------------------
    def _check_storage(self, probe: Job, result: PreflightResult) -> None:
        # Probe the store root itself (not a per-job directory) so a failed
        # probe can never leave a phantom job directory behind.
        probe_path = [".preflight"]
        try:
            self.storage.write({"ts": probe.created_ms}, *probe_path)
            self.storage.delete(*probe_path)
        except OSError as exc:
            result.add(ERROR, "data",
                       f"作业存储目录不可写 / job storage directory not writable: {exc}",
                       target=self.storage.path("jobs"))

    # ------------------------------------------------------------------
    def _finalize(self, result: PreflightResult) -> None:
        result.ok = not result.errors


def _find_cycle_touching(graph: dict[str, list[str]],
                         roots: set[str]) -> Optional[list[str]]:
    """Return a cycle path reachable from ``roots`` (incl. self-loops)."""
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {}
    stack: list[str] = []

    def dfs(node: str) -> Optional[list[str]]:
        color[node] = GREY
        stack.append(node)
        for nxt in graph.get(node, []):
            if color.get(nxt, WHITE) == GREY:
                idx = stack.index(nxt) if nxt in stack else 0
                return stack[idx:] + [nxt]
            if color.get(nxt, WHITE) == WHITE:
                found = dfs(nxt)
                if found:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for root in roots:
        found = dfs(root)
        if found:
            return found
    return None
