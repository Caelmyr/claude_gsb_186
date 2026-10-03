"""Tests for the pre-submission preflight (dry-run) check."""

import shutil
import tempfile
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.preflight import FAIL, PASS, WARN, Preflight, PreflightFailure


def base_payload(**overrides):
    p = {
        "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
        "num_map_tasks": 4, "num_reduce_tasks": 2, "input_rows": 800, "params": {},
    }
    p.update(overrides)
    return p


def codes(report, status=None):
    return {c["code"] for c in report["checks"]
            if status is None or c["status"] == status}


class TestPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.jm = JobManager(self.storage, ClusterConfig(), LogBus(self.storage))
        self.pf: Preflight = self.jm.preflight

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- healthy -----------------------------------------------------------
    def test_healthy_job_passes(self):
        r = self.pf.run(base_payload())
        self.assertTrue(r["ok"])
        self.assertEqual(r["failed"], 0)
        for cat in ("params", "data", "shape", "sharding", "dependencies"):
            self.assertTrue(any(c["category"] == cat for c in r["checks"]), cat)
        self.assertEqual(r["plan"]["effective_map_tasks"], 4)
        self.assertEqual(r["plan"]["generated_records"], 799)

    def test_preflight_does_not_persist_anything(self):
        before = list(self.storage.files("jobs"))
        self.pf.run(base_payload())
        after = list(self.storage.files("jobs"))
        self.assertEqual(before, after)

    # -- parameters --------------------------------------------------------
    def test_unknown_mapper_and_reducer_fail_and_locate_field(self):
        r = self.pf.run(base_payload(mapper="nope", reducer="ghost"))
        self.assertFalse(r["ok"])
        fails = [c for c in r["checks"] if c["status"] == FAIL]
        fields = {c["target"].get("field") for c in fails}
        self.assertIn("mapper", fields)
        self.assertIn("reducer", fields)

    def test_non_integer_counts_fail(self):
        r = self.pf.run(base_payload(num_map_tasks="abc", num_reduce_tasks=0,
                                     input_rows=-3))
        self.assertFalse(r["ok"])
        self.assertEqual(codes(r, FAIL),
                         {"num_map_invalid", "num_reduce_invalid", "input_rows_invalid"})

    def test_float_fraction_counts_fail(self):
        r = self.pf.run(base_payload(num_map_tasks=2.5))
        self.assertIn("num_map_invalid", codes(r, FAIL))

    def test_params_must_be_object(self):
        r = self.pf.run(base_payload(params=["not", "a", "dict"]))
        self.assertIn("params_invalid", codes(r, FAIL))

    def test_depends_on_must_be_list(self):
        r = self.pf.run(base_payload(depends_on="job-x"))
        self.assertIn("depends_on_invalid", codes(r, FAIL))

    def test_grep_pattern_warnings(self):
        r = self.pf.run(base_payload(mapper="grep_mapper", num_reduce_tasks=1,
                                     input_rows=200))
        self.assertIn("grep_pattern_default", codes(r, WARN))
        r2 = self.pf.run(base_payload(mapper="grep_mapper", num_reduce_tasks=1,
                                      input_rows=200, params={"pattern": "  "}))
        self.assertIn("grep_pattern_empty", codes(r2, WARN))

    def test_fault_injection_warns(self):
        r = self.pf.run(base_payload(params={"simulate_failure": True}))
        self.assertIn("fault_injection_on", codes(r, WARN))
        self.assertTrue(r["ok"])  # warnings never block on their own

    # -- data / empty ------------------------------------------------------
    def test_zero_rows_rejected_before_empty_dataset_is_built(self):
        r = self.pf.run(base_payload(input_rows=0))
        self.assertFalse(r["ok"])
        self.assertIn("input_rows_invalid", codes(r, FAIL))
        target = next(c["target"] for c in r["checks"] if c["code"] == "input_rows_invalid")
        self.assertEqual(target["field"], "input_rows")

    def test_declared_vs_effective_size_mismatch_warns(self):
        # Text generation yields rows-1 records; a 2-row declaration expands to
        # one record, a 50% shortfall worth surfacing before the run.
        r = self.pf.run(base_payload(input_rows=2, num_map_tasks=1))
        self.assertIn("input_size_mismatch", codes(r, WARN))
        chk = next(c for c in r["checks"] if c["code"] == "input_size_mismatch")
        self.assertEqual(chk["target"]["declared"], 2)
        self.assertEqual(chk["target"]["actual"], 1)

    # -- shape -------------------------------------------------------------
    def test_grep_zero_match_warns_and_locates_pattern_field(self):
        r = self.pf.run(base_payload(
            mapper="grep_mapper", num_reduce_tasks=1, input_rows=200,
            params={"pattern": "definitely_no_such_token_xyz"}))
        self.assertIn("grep_no_match", codes(r, WARN))
        chk = next(c for c in r["checks"] if c["code"] == "grep_no_match")
        self.assertEqual(chk["target"]["field"], "params.pattern")

    def test_kv_shape_contract_uses_kv_input(self):
        # The built-in kv generator yields dict records -> kv_mapper passes.
        r = self.pf.run(base_payload(mapper="kv_mapper", reducer="sum_reducer",
                                     num_reduce_tasks=2, input_rows=300))
        self.assertTrue(r["ok"])

    # -- sharding ----------------------------------------------------------
    def test_map_task_clamp_warns(self):
        r = self.pf.run(base_payload(num_map_tasks=100, input_rows=30))
        self.assertIn("map_clamped", codes(r, WARN))
        chk = next(c for c in r["checks"] if c["code"] == "map_clamped")
        self.assertEqual(chk["target"]["requested"], 100)
        self.assertEqual(chk["target"]["effective"], 29)
        self.assertEqual(r["plan"]["effective_map_tasks"], 29)

    def test_too_many_reduce_tasks_warns_about_empty_partitions(self):
        r = self.pf.run(base_payload(num_map_tasks=2, num_reduce_tasks=20,
                                     input_rows=40))
        self.assertIn("reduce_partition_empty", codes(r, WARN))
        chk = next(c for c in r["checks"] if c["code"] == "reduce_partition_empty")
        self.assertIn("empty_partitions", chk["target"])

    def test_well_shaped_job_has_no_sharding_warnings(self):
        r = self.pf.run(base_payload(num_map_tasks=4, num_reduce_tasks=2,
                                     input_rows=800))
        self.assertIn("map_count_ok", codes(r, PASS))
        self.assertFalse(codes(r, WARN) & {"map_clamped", "shard_skew",
                                           "reduce_partition_empty"})

    # -- dependencies ------------------------------------------------------
    def test_missing_dependency_fails(self):
        r = self.pf.run(base_payload(depends_on=["job-does-not-exist"]))
        self.assertIn("dependency_missing", codes(r, FAIL))
        chk = next(c for c in r["checks"] if c["code"] == "dependency_missing")
        self.assertEqual(chk["target"]["job_id"], "job-does-not-exist")
        self.assertEqual(chk["target"]["index"], 0)

    def test_running_dependency_blocks_but_succeeded_passes(self):
        ok_dep = self.jm.submit(base_payload(name="dep-ok", input_rows=100))
        self.jm.set_job_status(ok_dep, C.JOB_SUCCEEDED)
        running_dep = self.jm.submit(base_payload(name="dep-run", input_rows=100))
        self.assertEqual(running_dep.status, C.JOB_MAP)

        r = self.pf.run(base_payload(depends_on=[ok_dep.job_id]))
        self.assertTrue(r["ok"])
        self.assertIn("dependency_ready", codes(r, PASS))

        r = self.pf.run(base_payload(depends_on=[ok_dep.job_id, running_dep.job_id]))
        self.assertIn("dependency_pending", codes(r, FAIL))

        self.jm.set_job_status(running_dep, C.JOB_FAILED)
        r = self.pf.run(base_payload(depends_on=[running_dep.job_id]))
        self.assertIn("dependency_terminal_bad", codes(r, FAIL))

    # -- submit gating -----------------------------------------------------
    def test_submit_blocked_by_preflight_failure(self):
        with self.assertRaises(PreflightFailure) as ctx:
            self.jm.submit(base_payload(mapper="nope"))
        self.assertFalse(ctx.exception.report["ok"])

    def test_submit_blocked_by_unfinished_dependency(self):
        dep = self.jm.submit(base_payload(name="dep", input_rows=100))  # stays MAP
        with self.assertRaises(PreflightFailure):
            self.jm.submit(base_payload(depends_on=[dep.job_id]))

    def test_submit_persists_depends_on_when_ready(self):
        dep = self.jm.submit(base_payload(name="dep", input_rows=100))
        self.jm.set_job_status(dep, C.JOB_SUCCEEDED)
        job = self.jm.submit(base_payload(depends_on=[dep.job_id]))
        self.assertEqual(job.depends_on, [dep.job_id])
        self.assertEqual(self.jm.get_job(job.job_id).depends_on, [dep.job_id])


if __name__ == "__main__":
    unittest.main()
