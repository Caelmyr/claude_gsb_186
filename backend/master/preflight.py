"""Pre-submission dry run ("preflight") for jobs.

Submitting a job used to start computation immediately, so a bad parameter, an
empty dataset, an unreadable shape or an unfinished prerequisite job was only
discovered halfway through execution.  ``Preflight`` performs the most complete
*read-only* health check possible **before** anything is scheduled:

1. **params** — every submitted field parses and lies in a legal range, and the
   referenced mapper/reducer are registered;
2. **data** — the input that *would* be generated/read exists and is non-empty,
   both as a whole and per shard;
3. **shape** — every input record satisfies the mapper's data contract, and the
   mapper/reducer actually run on a sample (emitting well-formed pairs);
4. **sharding** — the requested map/reduce task counts are feasible for this
   dataset (clamping, skew, empty reduce partitions);
5. **dependencies** — every job named in ``depends_on`` exists and has reached
   a state from which this job is allowed to start;
6. **cluster** — at least one live worker is available to run the tasks.

Nothing here writes shard files or dispatches a single task: the planner's
:meth:`ShardPlanner.build_inputs` is used instead of ``plan``.  Every finding is
a structured :class:`CheckResult` that locates the offending **field** (e.g.
``num_map_tasks``) or **data item** (shard id + record index), so the UI can
present all problems at once.  A single hard failure (``status == "fail"``)
blocks submission; warnings require explicit user acknowledgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.common import constants as C
from backend.common.hashing import partition_for
from backend.common.ids import shard_id
from backend.common.models import Job
from backend.tasks.registry import get_mapper, get_reducer, has_mapper, has_reducer
from backend.tasks.samples import input_kind_for

# ---------------------------------------------------------------------------
# Check statuses / categories
# ---------------------------------------------------------------------------
PASS = "pass"
WARN = "warn"
FAIL = "fail"

CATEGORIES = ("params", "data", "shape", "sharding", "dependencies", "cluster")
CATEGORY_LABELS = {
    "params": "参数合法性 Parameters",
    "data": "输入数据 Input data",
    "shape": "形状匹配 Shape match",
    "sharding": "切分策略 Sharding",
    "dependencies": "前置作业 Dependencies",
    "cluster": "集群就绪 Cluster readiness",
}

# At most this many records are fed through the mapper during a dry run.  The
# sample is spread evenly over the whole input so skew at either end is caught.
SAMPLE_SIZE = 1000
# Report at most this many offending record indices per check (the full count
# is always included).
MAX_SAMPLE_INDEXES = 10

# Per-mapper input contract: what shape the input records must have.
#   {"kind": "text"}              — strings (or {"text"|"line": str} dicts)
#   {"kind": "kv", "value": ...}  — dicts carrying a key and a numeric value
KV_MAPPERS = {"kv_mapper"}


@dataclass
class CheckResult:
    code: str
    category: str
    status: str                       # PASS | WARN | FAIL
    title: str
    message: str
    target: dict = field(default_factory=dict)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "status": self.status,
            "title": self.title,
            "message": self.message,
            "target": self.target,
            "suggestion": self.suggestion,
        }


class PreflightFailure(ValueError):
    """Raised by JobManager.submit when the preflight dry run has failures."""

    def __init__(self, report: dict) -> None:
        self.report = report
        first = next((c for c in report.get("checks", []) if c["status"] == FAIL), None)
        super().__init__(first["message"] if first else "preflight checks failed")


# ---------------------------------------------------------------------------
class Preflight:
    def __init__(self, storage, config, planner, job_manager=None, registry=None) -> None:
        self.storage = storage
        self.config = config
        self.planner = planner
        self.job_manager = job_manager      # used to resolve prerequisite jobs
        self.registry = registry            # used for the cluster-readiness check

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, payload: dict) -> dict:
        """Validate a job-submission payload without executing or persisting."""
        payload = payload if isinstance(payload, dict) else {}
        checks: list[CheckResult] = []

        fields = self._check_params(payload, checks)
        self._check_fault_injection(fields["params"], checks)

        # The data/shape/sharding checks all derive from the planned inputs;
        # they only make sense once mapper + core counts are coherent enough
        # to build a provisional Job.
        built: Optional[dict] = None
        if fields["mapper_ok"] and fields["reducer_ok"] and fields["rows_ok"] \
                and fields["map_ok"] and fields["reduce_ok"]:
            job = Job(
                job_id="preflight",
                name=fields["name"],
                mapper=fields["mapper"],
                reducer=fields["reducer"],
                num_map_tasks=fields["num_map"],
                num_reduce_tasks=fields["num_reduce"],
                input_rows=fields["input_rows"],
                params=fields["params"],
            )
            try:
                built = self.planner.build_inputs(job)
            except Exception as exc:  # noqa: BLE001 - a crash here must be a finding
                checks.append(CheckResult(
                    "data_build_failed", "data", FAIL,
                    "输入生成失败 Input generation failed",
                    f"为该作业构造输入时抛出异常: {type(exc).__name__}: {exc}",
                    target={"field": "input"},
                    suggestion="检查 input_rows / mapper 组合是否受支持。",
                ))
                built = None

            if built is not None:
                self._check_data(fields, built, checks)
                self._check_sharding(fields, job, built, checks)
                self._check_shape(job, fields, built, checks)

        self._check_dependencies(fields, checks)
        self._check_cluster(checks)

        return self._report(fields, built, checks)

    # ------------------------------------------------------------------
    # 1. Parameters
    # ------------------------------------------------------------------
    def _parse_int(self, raw: Any, default: int) -> tuple[int, bool]:
        """Return (value, ok).

        A present-but-unparseable value is a failure.  An *absent* field falls
        back to ``default`` (which the server injects from saved job defaults);
        an explicitly empty field is treated as invalid so the UI surfaces the
        empty box instead of silently running a 12k-row job.
        """
        if raw is None:
            return default, True
        if raw == "":
            return default, False
        if isinstance(raw, bool):
            return int(raw), True
        if isinstance(raw, float):
            return (int(raw), True) if raw.is_integer() else (default, False)
        try:
            return int(raw), True
        except (TypeError, ValueError):
            return default, False

    def _check_params(self, payload: dict, checks: list[CheckResult]) -> dict:
        defaults = payload.get("_defaults") or {}

        name = str(payload.get("name") or "untitled").strip() or "untitled"
        mapper = str(payload.get("mapper") or "")
        reducer = str(payload.get("reducer") or "")

        mapper_ok = has_mapper(mapper) if mapper else False
        if not mapper:
            checks.append(CheckResult(
                "mapper_missing", "params", FAIL,
                "缺少 Mapper", "mapper 字段为空。",
                target={"field": "mapper"}, suggestion="选择一个已注册的 map 函数。"))
        elif not mapper_ok:
            checks.append(CheckResult(
                "mapper_unknown", "params", FAIL,
                "未知 Mapper", f"mapper {mapper!r} 未在函数注册表中找到。",
                target={"field": "mapper", "value": mapper},
                suggestion="使用 /api/functions 列出的 mapper 名称。"))
        else:
            checks.append(CheckResult(
                "mapper_ok", "params", PASS, "Mapper",
                f"mapper {mapper!r} 可用。", target={"field": "mapper", "value": mapper}))

        reducer_ok = has_reducer(reducer) if reducer else False
        if not reducer:
            checks.append(CheckResult(
                "reducer_missing", "params", FAIL,
                "缺少 Reducer", "reducer 字段为空。",
                target={"field": "reducer"}, suggestion="选择一个已注册的 reduce 函数。"))
        elif not reducer_ok:
            checks.append(CheckResult(
                "reducer_unknown", "params", FAIL,
                "未知 Reducer", f"reducer {reducer!r} 未在函数注册表中找到。",
                target={"field": "reducer", "value": reducer},
                suggestion="使用 /api/functions 列出的 reducer 名称。"))
        else:
            checks.append(CheckResult(
                "reducer_ok", "params", PASS, "Reducer",
                f"reducer {reducer!r} 可用。", target={"field": "reducer", "value": reducer}))

        num_map, map_ok = self._parse_int(
            payload.get("num_map_tasks"), int(defaults.get("num_map_tasks", 8)))
        if not map_ok or num_map < 1:
            checks.append(CheckResult(
                "num_map_invalid", "params", FAIL, "Map 任务数非法",
                f"num_map_tasks={payload.get('num_map_tasks')!r} 不是 >= 1 的整数。",
                target={"field": "num_map_tasks", "value": payload.get("num_map_tasks")},
                suggestion="设置一个正整数（建议 1–1000）。"))
            map_ok = False
        else:
            checks.append(CheckResult(
                "num_map_ok", "params", PASS, "Map 任务数",
                f"请求 {num_map} 个 map 任务。",
                target={"field": "num_map_tasks", "value": num_map}))

        num_reduce, reduce_ok = self._parse_int(
            payload.get("num_reduce_tasks"), int(defaults.get("num_reduce_tasks", 4)))
        if not reduce_ok or num_reduce < 1:
            checks.append(CheckResult(
                "num_reduce_invalid", "params", FAIL, "Reduce 任务数非法",
                f"num_reduce_tasks={payload.get('num_reduce_tasks')!r} 不是 >= 1 的整数。",
                target={"field": "num_reduce_tasks", "value": payload.get("num_reduce_tasks")},
                suggestion="设置一个正整数（建议 1–500）。"))
            reduce_ok = False
        else:
            checks.append(CheckResult(
                "num_reduce_ok", "params", PASS, "Reduce 任务数",
                f"请求 {num_reduce} 个 reduce 任务。",
                target={"field": "num_reduce_tasks", "value": num_reduce}))

        input_rows, rows_ok = self._parse_int(
            payload.get("input_rows"), int(defaults.get("input_rows", 12000)))
        if not rows_ok or input_rows < 1:
            checks.append(CheckResult(
                "input_rows_invalid", "params", FAIL, "输入行数非法",
                f"input_rows={payload.get('input_rows')!r} 不是 >= 1 的整数。",
                target={"field": "input_rows", "value": payload.get("input_rows")},
                suggestion="设置一个正整数（界面最小值为 10）。"))
            rows_ok = False
        else:
            checks.append(CheckResult(
                "input_rows_ok", "params", PASS, "输入行数",
                f"声明输入 {input_rows} 行。",
                target={"field": "input_rows", "value": input_rows}))

        raw_params = payload.get("params") if "params" in payload else {}
        params_ok = isinstance(raw_params, dict)
        if not params_ok:
            checks.append(CheckResult(
                "params_invalid", "params", FAIL, "params 非法",
                f"params 必须是对象 (object)，实际为 {type(raw_params).__name__}。",
                target={"field": "params", "value": str(raw_params)[:80]},
                suggestion="以键值对象形式传递 mapper/reducer 参数。"))
            raw_params = {}
        params = dict(raw_params)
        if mapper_ok:
            params["input_kind"] = input_kind_for(mapper)
            self._check_mapper_params(mapper, params, checks)

        depends_on = payload.get("depends_on", [])
        if depends_on is not None and not isinstance(depends_on, list):
            checks.append(CheckResult(
                "depends_on_invalid", "dependencies", FAIL, "依赖声明非法",
                f"depends_on 必须是作业 id 的数组，实际为 {type(depends_on).__name__}。",
                target={"field": "depends_on", "value": str(depends_on)[:80]},
                suggestion="传一个作业 id 列表，或省略该字段。"))
            depends_on = []
        depends_on = [str(d).strip() for d in (depends_on or []) if str(d).strip()]

        return {
            "name": name, "mapper": mapper, "reducer": reducer,
            "mapper_ok": mapper_ok, "reducer_ok": reducer_ok,
            "num_map": num_map, "map_ok": map_ok,
            "num_reduce": num_reduce, "reduce_ok": reduce_ok,
            "input_rows": input_rows, "rows_ok": rows_ok,
            "params": params, "depends_on": depends_on,
        }

    def _check_mapper_params(self, mapper: str, params: dict,
                             checks: list[CheckResult]) -> None:
        if mapper == "grep_mapper":
            pattern = params.get("pattern", None)
            if pattern is None:
                checks.append(CheckResult(
                    "grep_pattern_default", "params", WARN, "Grep 未指定 pattern",
                    "params.pattern 缺省，mapper 将使用默认值 'map'。",
                    target={"field": "params.pattern"},
                    suggestion="如需检索其它关键词，请显式设置 params.pattern。"))
            elif not isinstance(pattern, str) or not pattern.strip():
                checks.append(CheckResult(
                    "grep_pattern_empty", "params", WARN, "Grep pattern 为空",
                    "params.pattern 为空字符串，将匹配每一条记录。",
                    target={"field": "params.pattern", "value": pattern},
                    suggestion="设置非空关键词，或确认全量匹配是预期行为。"))
            else:
                checks.append(CheckResult(
                    "grep_pattern_ok", "params", PASS, "Grep pattern",
                    f"将检索关键词 {pattern!r}。",
                    target={"field": "params.pattern", "value": pattern}))

    def _check_fault_injection(self, params: dict, checks: list[CheckResult]) -> None:
        if params.get("simulate_failure"):
            checks.append(CheckResult(
                "fault_injection_on", "params", WARN, "故障注入已开启",
                "simulate_failure=true：每个任务的首轮尝试将刻意失败并触发自动重试。",
                target={"field": "params.simulate_failure", "value": True},
                suggestion="仅用于演示容错；正式计算请关闭。"))

    # ------------------------------------------------------------------
    # 2. Input data existence / non-emptiness
    # ------------------------------------------------------------------
    def _check_data(self, fields: dict, built: dict, checks: list[CheckResult]) -> None:
        records = built["records"]
        total = len(records)
        if total == 0:
            checks.append(CheckResult(
                "input_empty", "data", FAIL, "输入数据为空",
                f"input_rows={fields['input_rows']} 经输入源展开后得到 0 条记录，"
                "map 阶段没有任何可处理的数据。",
                target={"field": "input_rows", "value": fields["input_rows"]},
                suggestion="增大 input_rows（文本输入至少需要 2 行）。"))
            return

        nulls = [i for i, r in enumerate(records) if r is None]
        if nulls:
            checks.append(CheckResult(
                "input_null_records", "data", FAIL, "存在空记录 (null)",
                f"{len(nulls)} 条输入记录为 null。",
                target={"field": "input.records", "record_indexes": nulls[:MAX_SAMPLE_INDEXES],
                        "count": len(nulls)},
                suggestion="移除或修复这些记录后再提交。"))

        chunks = built["chunks"]
        empty_shards = [i for i, c in enumerate(chunks) if len(c) == 0]
        if empty_shards:
            checks.append(CheckResult(
                "shard_empty", "data", FAIL, "存在空输入分片",
                f"{len(empty_shards)} 个 map 分片没有任何记录。",
                target={"field": "num_map_tasks",
                        "shard_indexes": empty_shards[:MAX_SAMPLE_INDEXES],
                        "shard_ids": [shard_id("in", i) for i in empty_shards[:MAX_SAMPLE_INDEXES]]},
                suggestion="减少 map 任务数或增加输入数据。"))

        if not nulls and not empty_shards:
            checks.append(CheckResult(
                "input_ok", "data", PASS, "输入数据存在且非空",
                f"共生成 {total} 条记录，分布在 {len(chunks)} 个输入分片中。",
                target={"field": "input", "records": total, "shards": len(chunks)}))

        # Declared-vs-effective size: input_rows is the user's statement about
        # the dataset; the planner's expansion is what the job will actually run
        # on.  A material discrepancy means the job is not the one the user
        # pictured (e.g. an external/empty source clamped to one filler record).
        declared = max(1, int(fields["input_rows"]))
        if total and abs(total - declared) / declared > 0.1:
            checks.append(CheckResult(
                "input_size_mismatch", "data", WARN, "声明行数与实际数据量不符",
                f"input_rows 声明 {declared} 行，但输入源实际展开为 {total} 条记录"
                f"（偏差 {abs(total - declared) / declared * 100:.0f}%）。",
                target={"field": "input_rows", "declared": declared, "actual": total},
                suggestion="核对数据来源；文本生成器的有效记录数为 input_rows-1。"))

    # ------------------------------------------------------------------
    # 3. Shape match + mapper/reducer dry run on a sample
    # ------------------------------------------------------------------
    def _sample_indexes(self, n: int) -> list[int]:
        if n <= SAMPLE_SIZE:
            return list(range(n))
        step = n / SAMPLE_SIZE
        return [int(i * step) for i in range(SAMPLE_SIZE)]

    def _shard_offsets(self, chunks: list[list]) -> list[int]:
        offsets, total = [], 0
        for c in chunks:
            offsets.append(total)
            total += len(c)
        return offsets

    def _shard_for(self, offsets: list[int], record_index: int) -> str:
        shard_index = 0
        for i, start in enumerate(offsets):
            if start <= record_index:
                shard_index = i
            else:
                break
        return shard_id("in", shard_index)

    def _check_shape(self, job: Job, fields: dict, built: dict,
                     checks: list[CheckResult]) -> None:
        records, chunks = built["records"], built["chunks"]
        if not records:
            return
        offsets = self._shard_offsets(chunks)
        mapper_name, reducer_name = fields["mapper"], fields["reducer"]
        try:
            mapper_fn = get_mapper(mapper_name)
        except KeyError:
            return
        kv_mode = mapper_name in KV_MAPPERS

        # --- static contract over the (sampled) records ----------------
        sample_idx = self._sample_indexes(len(records))
        bad_shape: list[dict] = []
        for gi in sample_idx:
            rec = records[gi]
            reason = self._shape_violation(kv_mode, rec)
            if reason:
                bad_shape.append({"record_index": gi, "shard_id": self._shard_for(offsets, gi),
                                  "reason": reason, "record": self._preview(rec)})
        if bad_shape:
            checks.append(CheckResult(
                "shape_mismatch", "shape", FAIL,
                "输入形状与 mapper 不匹配",
                f"mapper {mapper_name!r} 期望{'键值对象 {key, value}' if kv_mode else '文本行'}，"
                f"抽样 {len(sample_idx)} 条中有 {len(bad_shape)} 条不满足契约。",
                target={"field": "input.records", "mapper": mapper_name,
                        "expected": "kv" if kv_mode else "text",
                        "offenders": bad_shape[:MAX_SAMPLE_INDEXES]},
                suggestion="更换 mapper，或修复/转换这些输入记录。"))

        # --- run the mapper per record, then the reducer per key -------
        groups: dict[Any, list] = {}
        bad_run: list[dict] = []
        bad_pair: list[dict] = []
        non_numeric: list[dict] = []
        matched_records = 0
        for gi in sample_idx:
            rec = records[gi]
            try:
                pairs = mapper_fn([rec], fields["params"])
            except Exception as exc:  # noqa: BLE001
                bad_run.append({"record_index": gi, "shard_id": self._shard_for(offsets, gi),
                                "record": self._preview(rec),
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not isinstance(pairs, (list, tuple)):
                bad_pair.append({"record_index": gi, "reason": "mapper 返回值不是列表",
                                 "got": type(pairs).__name__})
                continue
            if pairs:
                matched_records += 1
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    bad_pair.append({"record_index": gi,
                                     "shard_id": self._shard_for(offsets, gi),
                                     "reason": "输出不是 (key, value) 二元组",
                                     "got": self._preview(pair)})
                    continue
                key, value = pair
                try:
                    partition_for(key, fields["num_reduce"])
                except TypeError as exc:
                    bad_pair.append({"record_index": gi,
                                     "shard_id": self._shard_for(offsets, gi),
                                     "reason": f"key 无法哈希分区: {exc}",
                                     "got": self._preview(key)})
                    continue
                if mapper_name in KV_MAPPERS and not isinstance(value, (int, float)):
                    non_numeric.append({"record_index": gi, "shard_id": self._shard_for(offsets, gi),
                                        "value": self._preview(value)})
                groups.setdefault(key, []).append(value)

        if bad_run:
            checks.append(CheckResult(
                "mapper_runtime", "shape", FAIL, "Mapper 试运行失败",
                f"在抽样数据上执行 mapper 时有 {len(bad_run)} 条记录抛出异常。",
                target={"field": "mapper", "mapper": mapper_name,
                        "offenders": bad_run[:MAX_SAMPLE_INDEXES]},
                suggestion="根据报错修复输入记录或 params。"))
        if bad_pair:
            checks.append(CheckResult(
                "mapper_output", "shape", FAIL, "Mapper 输出结构非法",
                f"{len(bad_pair)} 个输出不是可分区的 (key, value) 二元组。",
                target={"field": "mapper", "offenders": bad_pair[:MAX_SAMPLE_INDEXES]},
                suggestion="mapper 必须返回 [(key, value), ...] 且 key 可哈希。"))
        if non_numeric:
            checks.append(CheckResult(
                "kv_value_non_numeric", "shape", WARN, "KV value 非数值",
                f"{len(non_numeric)} 条记录的 value 不是数值，聚合时将被记为 0。",
                target={"field": "input.records", "mapper": mapper_name,
                        "offenders": non_numeric[:MAX_SAMPLE_INDEXES]},
                suggestion="将 value 转换为数值，或确认按 0 聚合符合预期。"))

        # Reducer dry run over the sampled groups.
        if groups and fields["reducer_ok"]:
            try:
                reducer_fn = get_reducer(reducer_name)
            except KeyError:
                reducer_fn = None
            reducer_errors: list[dict] = []
            checked = 0
            for key, values in groups.items():
                if checked >= 200 or reducer_fn is None:
                    break
                try:
                    result = reducer_fn(key, values, fields["params"])
                except Exception as exc:  # noqa: BLE001
                    reducer_errors.append({"key": self._preview(key),
                                           "error": f"{type(exc).__name__}: {exc}"})
                checked += 1
            if reducer_errors:
                checks.append(CheckResult(
                    "reducer_runtime", "shape", FAIL, "Reducer 试运行失败",
                    f"在抽样分组上执行 reducer 时有 {len(reducer_errors)} 个 key 抛出异常。",
                    target={"field": "reducer", "reducer": reducer_name,
                            "offenders": reducer_errors[:MAX_SAMPLE_INDEXES]},
                    suggestion="根据报错检查 mapper 输出的 value 类型是否与 reducer 匹配。"))

        if not bad_shape and not bad_run and not bad_pair and not non_numeric:
            full = len(records) == len(sample_idx)
            scope = "全部记录" if full else f"均匀抽样 {len(sample_idx)} 条"
            if mapper_name == "grep_mapper":
                if matched_records == 0:
                    checks.append(CheckResult(
                        "grep_no_match", "shape", WARN, "Grep 零命中",
                        f"{scope}中没有任何记录包含 pattern="
                        f"{fields['params'].get('pattern', 'map')!r}，结果将为空。",
                        target={"field": "params.pattern",
                                "value": fields["params"].get("pattern", "map")},
                        suggestion="确认关键词拼写，或更换数据集。"))
                else:
                    checks.append(CheckResult(
                        "shape_ok", "shape", PASS, "形状与试运行通过",
                        f"{scope}通过 mapper 试运行，{matched_records} 条记录命中关键词。",
                        target={"field": "input.records"}))
            elif not groups:
                checks.append(CheckResult(
                    "mapper_zero_output", "shape", WARN, "Mapper 零输出",
                    f"{scope}未产生任何 (key, value)，作业结果将为空。",
                    target={"field": "input.records", "mapper": mapper_name},
                    suggestion="检查输入内容是否为 mapper 能解析的文本/对象。"))
            else:
                checks.append(CheckResult(
                    "shape_ok", "shape", PASS, "形状与试运行通过",
                    f"{scope}通过 mapper/reducer 试运行，"
                    f"产出 {len(groups)} 个不同 key、{sum(len(v) for v in groups.values())} 条键值对。",
                    target={"field": "input.records", "sampled_keys": len(groups)}))

    def _shape_violation(self, kv_mode: bool, rec: Any) -> str:
        if rec is None:
            return "记录为 null"
        if kv_mode:
            if not isinstance(rec, dict):
                return f"期望 dict，实际为 {type(rec).__name__}"
            if "key" not in rec:
                return "缺少 'key' 字段"
            if "value" not in rec:
                return "缺少 'value' 字段"
            try:
                float(rec["value"])
            except (TypeError, ValueError):
                return f"value={rec['value']!r} 不是数值"
            return ""
        # text contract: str, or a dict carrying a text/line field
        if isinstance(rec, str):
            return ""
        if isinstance(rec, dict) and ("text" in rec or "line" in rec):
            return ""
        if isinstance(rec, dict):
            return "dict 记录缺少 'text'/'line' 字段，将被强制转成字符串"
        return ""

    # ------------------------------------------------------------------
    # 4. Sharding feasibility
    # ------------------------------------------------------------------
    def _check_sharding(self, fields: dict, job: Job, built: dict,
                        checks: list[CheckResult]) -> None:
        requested_map = fields["num_map"]
        effective_map = len(built["chunks"])

        if effective_map < requested_map:
            checks.append(CheckResult(
                "map_clamped", "sharding", WARN, "Map 任务数将被收敛",
                f"请求 {requested_map} 个 map 任务，但只有 {len(built['records'])} 条输入记录；"
                f"实际将创建 {effective_map} 个（每个分片至少一条记录）。",
                target={"field": "num_map_tasks", "requested": requested_map,
                        "effective": effective_map},
                suggestion="减小 num_map_tasks，或增加输入数据。"))
        else:
            checks.append(CheckResult(
                "map_count_ok", "sharding", PASS, "Map 切分可行",
                f"{len(built['records'])} 条记录可均匀切分为 {effective_map} 个 map 任务。",
                target={"field": "num_map_tasks", "effective": effective_map}))

        sizes = [len(c) for c in built["chunks"]]
        if sizes and min(sizes) > 0:
            lo, hi = min(sizes), max(sizes)
            if hi > lo * 2 and hi - lo >= 8:
                checks.append(CheckResult(
                    "shard_skew", "sharding", WARN, "分片负载不均",
                    f"最大分片 {hi} 条，最小分片 {lo} 条，部分 map 任务会明显更慢。",
                    target={"field": "num_map_tasks", "min": lo, "max": hi,
                            "shard_sizes": sizes},
                    suggestion="调整 map 任务数使记录数可被较均匀地整除。"))
            elif effective_map == requested_map:
                checks.append(CheckResult(
                    "shard_balance_ok", "sharding", PASS, "分片均衡",
                    f"各分片记录数介于 {lo}–{hi} 之间。",
                    target={"field": "num_map_tasks", "min": lo, "max": hi}))

        # Reduce-partition feasibility can only be judged exactly when the
        # whole dataset was sampled; otherwise the planner reports an estimate.
        r = fields["num_reduce"]
        records = built["records"]
        full_sample = len(records) <= SAMPLE_SIZE
        if full_sample:
            histogram = {p: 0 for p in range(r)}
            keys: set = set()
            mapper_name = fields["mapper"]
            if has_mapper(mapper_name):
                mapper_fn = get_mapper(mapper_name)
                params = fields["params"]
                try:
                    for key, _value in mapper_fn(records, params):
                        keys.add(key)
                        histogram[partition_for(key, r)] += 1
                except Exception:  # noqa: BLE001 - already reported by shape checks
                    histogram = {}
            empty_parts = [p for p, n in histogram.items() if n == 0]
            if not keys:
                # Zero emitted keys are reported per-mapper by the shape check
                # (grep 零命中 / mapper 零输出); don't double-report the whole
                # reduce side as empty.
                pass
            elif empty_parts:
                checks.append(CheckResult(
                    "reduce_partition_empty", "sharding", WARN, "部分 Reduce 分区为空",
                    f"{len(empty_parts)}/{r} 个 reduce 分区收不到任何键（全量 {len(keys)} 个 key），"
                    "对应的 reduce 任务将空跑。",
                    target={"field": "num_reduce_tasks", "value": r,
                            "empty_partitions": empty_parts[:MAX_SAMPLE_INDEXES],
                            "distinct_keys": len(keys)},
                    suggestion="将 num_reduce_tasks 调小到不超过 key 的数量。"))
            else:
                checks.append(CheckResult(
                    "reduce_partition_ok", "sharding", PASS, "Reduce 分区可行",
                    f"{len(keys)} 个 key 经哈希分布到 {r} 个分区，无空分区。",
                    target={"field": "num_reduce_tasks", "value": r,
                            "partition_histogram": histogram}))

    # ------------------------------------------------------------------
    # 5. Prerequisite jobs
    # ------------------------------------------------------------------
    def _check_dependencies(self, fields: dict, checks: list[CheckResult]) -> None:
        deps = fields["depends_on"]
        if not deps:
            checks.append(CheckResult(
                "dependencies_none", "dependencies", PASS, "无前置作业",
                "该作业没有声明 depends_on，可独立启动。",
                target={"field": "depends_on"}))
            return

        if self.job_manager is None:
            return
        seen: set = set()
        for i, dep_id in enumerate(deps):
            if dep_id in seen:
                continue
            seen.add(dep_id)
            target = {"field": "depends_on", "index": i, "job_id": dep_id}
            dep = self.job_manager.get_job(dep_id)
            if dep is None:
                checks.append(CheckResult(
                    "dependency_missing", "dependencies", FAIL,
                    "前置作业不存在", f"depends_on[{i}] 引用的作业 {dep_id!r} 不存在。",
                    target=target, suggestion="核对作业 id，或先提交该前置作业。"))
            elif dep.status in (C.JOB_FAILED, C.JOB_CANCELLED):
                checks.append(CheckResult(
                    "dependency_terminal_bad", "dependencies", FAIL,
                    "前置作业未成功",
                    f"前置作业 {dep.name} ({dep_id}) 状态为 "
                    f"{C.state_label(dep.status)}，其输出不可用。",
                    target=target,
                    suggestion="重新运行前置作业直到成功，或移除该依赖。"))
            elif dep.status == C.JOB_SUCCEEDED:
                checks.append(CheckResult(
                    "dependency_ready", "dependencies", PASS,
                    "前置作业已完成", f"前置作业 {dep.name} ({dep_id}) 已成功，输出可用。",
                    target=target))
            else:
                checks.append(CheckResult(
                    "dependency_pending", "dependencies", FAIL,
                    "前置作业尚未完成",
                    f"前置作业 {dep.name} ({dep_id}) 仍处于 {C.state_label(dep.status)}，"
                    "本作业暂不具备启动条件。",
                    target=target,
                    suggestion="等待前置作业成功后再提交，或稍后重试预检。"))

    # ------------------------------------------------------------------
    # 6. Cluster readiness
    # ------------------------------------------------------------------
    def _check_cluster(self, checks: list[CheckResult]) -> None:
        if self.registry is None:
            return
        alive = self.registry.alive()
        if not alive:
            checks.append(CheckResult(
                "cluster_no_workers", "cluster", WARN, "没有存活的 Worker",
                "当前集群没有任何存活 worker；作业可以提交，但任务会停留在 PENDING "
                "直到有 worker 注册。",
                target={"field": "cluster.workers", "alive": 0},
                suggestion="先启动至少一个 worker，再提交作业。"))
        else:
            busy = sum(w.running_tasks + w.queued_tasks for w in alive)
            checks.append(CheckResult(
                "cluster_ready", "cluster", PASS, "集群就绪",
                f"{len(alive)} 个存活 worker 可执行任务（在途任务 {busy} 个）。",
                target={"field": "cluster.workers", "alive": len(alive), "in_flight": busy}))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _report(self, fields: dict, built: Optional[dict],
                checks: list[CheckResult]) -> dict:
        serialized = [c.to_dict() for c in checks]
        counts = {PASS: 0, WARN: 0, FAIL: 0}
        for c in serialized:
            counts[c["status"]] += 1

        plan: dict[str, Any] = {
            "name": fields["name"],
            "mapper": fields["mapper"],
            "reducer": fields["reducer"],
            "input_kind": fields["params"].get("input_kind", ""),
            "requested_map_tasks": fields["num_map"],
            "requested_reduce_tasks": fields["num_reduce"],
            "declared_input_rows": fields["input_rows"],
            "depends_on": fields["depends_on"],
        }
        if built is not None:
            plan.update({
                "generated_records": len(built["records"]),
                "effective_map_tasks": len(built["chunks"]),
                "effective_reduce_tasks": len(built["reduce_tasks"]),
                "shard_sizes": [len(c) for c in built["chunks"]],
            })

        failures = [c for c in serialized if c["status"] == FAIL]
        return {
            "ok": not failures,
            "passed": counts[PASS],
            "warnings": counts[WARN],
            "failed": counts[FAIL],
            "checks": serialized,
            "plan": plan,
        }

    @staticmethod
    def _preview(value: Any, limit: int = 80) -> str:
        text = repr(value)
        return text if len(text) <= limit else text[:limit - 1] + "…"
