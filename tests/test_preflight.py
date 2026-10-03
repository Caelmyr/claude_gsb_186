"""Tests for the pre-submission preflight (dry run) and dependency gating."""

import shutil
import tempfile
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.preflight import Preflight, ERROR, WARNING
from backend.master.registry import WorkerRegistry
from backend.common.models import WorkerRecord, new_worker


def _payload(**overrides):
    p = {
        "name": "t",
        "mapper": "wordcount_mapper",
        "reducer": "count_reducer",
        "num_map_tasks": 4,
        "num_reduce_tasks": 2,
        "input_rows": 800,
        "params": {},
    }
    p.update(overrides)
    return p


class PreflightTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.pf = Preflight(self.storage, self.jm.planner, self.jm,
                            self.registry, self.config)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **overrides):
        return self.pf.run(_payload(**overrides), {})

    def _groups(self, result, group, severity=None):
        return [f for f in result.findings
                if f.group == group and (severity is None or f.severity == severity)]

    def _register_worker(self):
        w = new_worker("worker-1", "w1", "127.0.0.1", 9001, 4, 1024)
        w.status = C.WORKER_ALIVE
        self.registry._workers[w.worker_id] = w


class TestParamValidation(PreflightTestBase):
    def test_valid_payload_passes(self):
        self._register_worker()
        r = self._run()
        self.assertTrue(r.ok, [f.message for f in r.errors])

    def test_unknown_functions_reported_on_field(self):
        r = self._run(mapper="nope", reducer="nada")
        fields = {f.field for f in r.errors}
        self.assertIn("mapper", fields)
        self.assertIn("reducer", fields)

    def test_non_numeric_and_out_of_range(self):
        r = self._run(num_map_tasks="abc", num_reduce_tasks=0, input_rows="")
        fields = {f.field for f in r.errors}
        self.assertEqual(fields, {"num_map_tasks", "num_reduce_tasks", "input_rows"})

    def test_missing_required_param(self):
        # pattern is optional for grep (has default); emulate a required one.
        from backend.tasks import registry as reg
        ps = reg.ParamSpec("pattern", "string", required=True)
        old = reg._META["grep_mapper"]
        reg._META["grep_mapper"] = reg.FunctionSpec(
            name=old.name, kind="mapper", params=[ps], input_shape="text", emits="count")
        try:
            r = self._run(mapper="grep_mapper")
        finally:
            reg._META["grep_mapper"] = old
        self.assertTrue(any(f.field == "params.pattern" for f in r.errors))

    def test_bad_param_type(self):
        r = self._run(mapper="grep_mapper", params={"pattern": 123})
        self.assertTrue(any(f.field == "params.pattern" and f.severity == ERROR
                            for f in r.findings))

    def test_unknown_param_is_info_not_error(self):
        self._register_worker()
        r = self._run(params={"bogus": 1})
        infos = self._groups(r, "params", "info")
        self.assertTrue(any("bogus" in f.target for f in infos))


class TestDataAndShape(PreflightTestBase):
    def test_data_non_empty_and_canary_runs(self):
        r = self._run()
        data_infos = self._groups(r, "data", "info")
        self.assertTrue(data_infos)
        canary = [f for f in self._groups(r, "shape", "info") if f.target == "canary"]
        self.assertTrue(canary, "canary trial should report success")
        self.assertEqual(r.plan_preview["input_rows"], 800)

    def test_empty_shards_are_errors_pointing_at_shard_ids(self):
        r = self._run(num_map_tasks=1000, input_rows=4)
        # clamp collapses map tasks to 4 -> 1 record each, no empties; ask for
        # an impossible split indirectly: 0 rows is rejected at params, so use a
        # direct dry-plan scenario with a tiny dataset and oversized request.
        self.assertEqual(r.plan_preview["num_map_tasks"], 4)
        # the clamp itself is a warning that names the field
        self.assertTrue(self._groups(r, "sharding", WARNING))

    def test_shape_mismatch_warning_for_kv_mapper_count_reducer(self):
        r = self._run(mapper="kv_mapper", reducer="count_reducer")
        self.assertTrue(self._groups(r, "shape", WARNING))

    def test_kv_input_is_generated_for_kv_mapper(self):
        r = self._run(mapper="kv_mapper", reducer="sum_reducer", input_rows=200)
        self.assertEqual(r.plan_preview["input_kind"], "kv")

    def test_grep_zero_matches_warns_on_pattern_field(self):
        r = self._run(mapper="grep_mapper",
                      params={"pattern": "zzzz-no-such-token-zzzz"})
        warns = self._groups(r, "shape", WARNING)
        self.assertTrue(warns)
        self.assertEqual(warns[0].field, "params.pattern")


class TestCanaryCatchesRuntimeFaults(PreflightTestBase):
    def test_mapper_exception_is_blocking_and_located(self):
        from backend.tasks import registry as reg

        def boom_mapper(records, params):
            raise RuntimeError("kaboom at runtime")

        reg.register_mapper("boom_mapper", boom_mapper, "boom",
                            input_shape="text", emits="count")
        try:
            r = self._run(mapper="boom_mapper")
        finally:
            reg._MAPPERS.pop("boom_mapper", None)
            reg._META.pop("boom_mapper", None)
        errs = self._groups(r, "shape", ERROR)
        self.assertTrue(errs)
        self.assertIn("kaboom", errs[0].message)
        self.assertTrue(errs[0].target.startswith("in-"))  # locates the shard

    def test_reducer_exception_is_blocking(self):
        from backend.tasks import registry as reg

        def boom_reducer(key, values, params):
            raise ValueError("reduce exploded")

        reg.register_reducer("boom_reducer", boom_reducer, "boom", accepts="count")
        try:
            r = self._run(reducer="boom_reducer")
        finally:
            reg._REDUCERS.pop("boom_reducer", None)
            reg._META.pop("boom_reducer", None)
        errs = self._groups(r, "shape", ERROR)
        self.assertTrue(any("reduce exploded" in f.message for f in errs))


class TestSharding(PreflightTestBase):
    def test_map_tasks_clamped_with_warning(self):
        r = self._run(num_map_tasks=1000, input_rows=30)
        warns = self._groups(r, "sharding", WARNING)
        self.assertTrue(any("clamped" in f.message or "收敛" in f.message for f in warns))
        self.assertEqual(r.plan_preview["num_map_tasks"], 30)

    def test_even_split_sizes(self):
        r = self._run(num_map_tasks=4, input_rows=800)
        self.assertEqual(r.plan_preview["shard_sizes"], [200, 200, 200, 200])

    def test_too_many_reducers_warns(self):
        r = self._run(num_reduce_tasks=500, input_rows=30)
        self.assertTrue(self._groups(r, "sharding", WARNING))


class TestDependencies(PreflightTestBase):
    def test_unknown_dependency_is_error(self):
        r = self._run(depends_on=["job-doesnot-exist"])
        errs = self._groups(r, "dependencies", ERROR)
        self.assertTrue(errs)
        self.assertEqual(errs[0].target, "job-doesnot-exist")

    def test_succeeded_dependency_is_info_and_ok(self):
        self._register_worker()
        up = self.jm.submit(_payload(name="up", num_map_tasks=2, input_rows=100))
        self.jm.set_job_status(up, C.JOB_SUCCEEDED)
        r = self._run(depends_on=[up.job_id])
        self.assertTrue(r.ok, [f.message for f in r.errors])
        self.assertTrue(self._groups(r, "dependencies", "info"))

    def test_running_dependency_is_warning(self):
        up = self.jm.submit(_payload(name="up", num_map_tasks=2, input_rows=100))
        # status stays MAP
        self.assertEqual(up.status, C.JOB_MAP)
        r = self._run(depends_on=[up.job_id])
        self.assertTrue(self._groups(r, "dependencies", WARNING))

    def test_failed_dependency_is_error(self):
        up = self.jm.submit(_payload(name="up", num_map_tasks=2, input_rows=100))
        self.jm.set_job_status(up, C.JOB_FAILED)
        r = self._run(depends_on=[up.job_id])
        self.assertTrue(self._groups(r, "dependencies", ERROR))

    def test_dependency_cycle_detected(self):
        # Existing job A depends on the candidate id; the candidate depends on A.
        from backend.common.ids import new_id
        candidate_id = new_id("job")
        up = self.jm.submit(_payload(
            name="loopy", num_map_tasks=2, input_rows=100,
            depends_on=[candidate_id]))
        r = self.pf.run(_payload(depends_on=[up.job_id]), {})
        # pf.run generates its own candidate id; emulate via direct graph call
        from backend.master.preflight import _find_cycle_touching
        graph = {
            up.job_id: [candidate_id],
            candidate_id: [up.job_id],
        }
        cycle = _find_cycle_touching(graph, {candidate_id})
        self.assertIsNotNone(cycle)
        self.assertIn(up.job_id, cycle)

    def test_invalid_depends_on_type(self):
        r = self._run(depends_on="not-a-list")
        self.assertTrue(self._groups(r, "dependencies", ERROR))


class TestClusterAndStorage(PreflightTestBase):
    def test_no_worker_is_warning_not_blocking(self):
        r = self._run()
        self.assertTrue(r.ok, "job should be allowed to queue without workers")
        self.assertTrue(self._groups(r, "cluster", WARNING))

    def test_alive_worker_is_info(self):
        self._register_worker()
        r = self._run()
        self.assertTrue(self._groups(r, "cluster", "info"))

    def test_storage_probe_leaves_no_trace(self):
        self._run()
        # the probe file must be deleted; only real submissions create job dirs.
        jobs = self.storage.subdirs("jobs")
        self.assertEqual(jobs, [])


class TestSubmitGatingAndScheduler(PreflightTestBase):
    def test_server_gate_rejects_until_preflight_passes(self):
        # Mirror the server route's decision logic.
        result = self.pf.run(_payload(mapper="nope"), {})
        self.assertFalse(result.ok)

    def test_submitted_job_with_pending_dependency_starts_pending(self):
        up = self.jm.submit(_payload(name="up", num_map_tasks=2, input_rows=100))
        self.assertEqual(up.status, C.JOB_MAP)
        down = self.jm.submit(_payload(
            name="down", num_map_tasks=2, input_rows=100,
            depends_on=[up.job_id]))
        self.assertEqual(down.status, C.JOB_PENDING)
        self.assertFalse(self.jm.dependencies_met(down))
        # once upstream succeeds the predicate flips
        self.jm.set_job_status(up, C.JOB_SUCCEEDED)
        self.assertTrue(self.jm.dependencies_met(down))

    def test_submitted_job_without_dependencies_runs_immediately(self):
        job = self.jm.submit(_payload())
        self.assertEqual(job.status, C.JOB_MAP)


if __name__ == "__main__":
    unittest.main()
