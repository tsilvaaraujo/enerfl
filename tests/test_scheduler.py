import json

import pytest

from enerfl.scheduler import expand_slurm_nodelist, scheduler_info
from enerfl.server import Profiler


@pytest.mark.parametrize(
    "nodelist, expected",
    [
        ("node1", ["node1"]),
        ("node[1-3]", ["node1", "node2", "node3"]),
        ("node[01-03]", ["node01", "node02", "node03"]),
        ("node[08-10]", ["node08", "node09", "node10"]),
        ("node[1,3-4],gpu7", ["node1", "node3", "node4", "gpu7"]),
        ("a1,b[1-2],c3", ["a1", "b1", "b2", "c3"]),
        ("rack[1-2]-n[1-2]", ["rack1-n1", "rack1-n2", "rack2-n1", "rack2-n2"]),
        ("n[1-2]-ib", ["n1-ib", "n2-ib"]),
    ],
)
def test_expand_slurm_nodelist(nodelist, expected):
    assert expand_slurm_nodelist(nodelist) == expected


@pytest.mark.parametrize(
    "bad", ["node[1-3", "node1-3]", "node[a-b]", "node[3-1]", "node[1-]", "n[[1]]"]
)
def test_expand_slurm_nodelist_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        expand_slurm_nodelist(bad)


def test_expand_slurm_nodelist_refuses_absurd_sizes():
    with pytest.raises(ValueError):
        expand_slurm_nodelist("n[1-99999]-m[1-99999]")


def test_no_scheduler_means_none():
    assert scheduler_info({"HOME": "/x"}) is None


def test_slurm_job():
    info = scheduler_info(
        {
            "SLURM_JOB_ID": "4242",
            "SLURM_JOB_NAME": "fl-train",
            "SLURM_JOB_PARTITION": "normal-arm",
            "SLURM_JOB_ACCOUNT": "proj1",
            "SLURM_CLUSTER_NAME": "deucalion",
            "SLURM_JOB_NODELIST": "cn[0101-0102],cn0200",
            "SLURM_JOB_NUM_NODES": "3",
            "SLURM_NODEID": "1",
            "SLURM_PROCID": "1",
            "SLURM_CPUS_ON_NODE": "48",
            "SLURM_ARRAY_JOB_ID": "4000",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_JOB_USER": "someone",
            "SLURM_SUBMIT_DIR": "/home/someone/secret",
        }
    )
    assert info["name"] == "slurm"
    assert info["job_id"] == "4242"
    assert info["partition"] == "normal-arm"
    assert info["cluster"] == "deucalion"
    assert info["node_list_raw"] == "cn[0101-0102],cn0200"
    assert info["nodes"] == ["cn0101", "cn0102", "cn0200"]
    assert info["num_nodes"] == 3
    assert (info["node_id"], info["proc_id"], info["cpus_on_node"]) == (1, 1, 48)
    assert (info["array_job_id"], info["array_task_id"]) == ("4000", "7")
    assert "someone" not in json.dumps(info)


def test_slurm_keeps_the_raw_list_when_it_cannot_be_expanded():
    info = scheduler_info(
        {"SLURM_JOB_ID": "1", "SLURM_JOB_NODELIST": "node[1-"}
    )
    assert info["node_list_raw"] == "node[1-"
    assert info["nodes"] is None


def test_slurm_tolerates_missing_and_bad_numbers():
    info = scheduler_info({"SLURM_JOB_ID": "1", "SLURM_NODEID": "x"})
    assert info["node_id"] is None
    assert info["num_nodes"] is None
    assert info["nodes"] is None


def test_oar_job_reads_the_node_file(tmp_path):
    node_file = tmp_path / "nodes"
    node_file.write_text("gros-1\ngros-1\ngros-2\ngros-2\n\n")
    info = scheduler_info(
        {
            "OAR_JOB_ID": "99",
            "OAR_JOB_NAME": "fl",
            "OAR_NODEFILE": str(node_file),
        }
    )
    assert info["name"] == "oar"
    assert info["job_id"] == "99"
    assert info["nodes"] == ["gros-1", "gros-2"]
    assert info["slots_per_node"] == {"gros-1": 2, "gros-2": 2}
    assert info["num_nodes"] == 2


def test_oar_with_unreadable_node_file(tmp_path):
    info = scheduler_info(
        {"OAR_JOB_ID": "99", "OAR_NODEFILE": str(tmp_path / "missing")}
    )
    assert info["nodes"] is None and info["num_nodes"] is None


def test_slurm_wins_when_both_are_present():
    assert scheduler_info({"SLURM_JOB_ID": "1", "OAR_JOB_ID": "2"})["name"] == "slurm"


def test_profiler_metadata_includes_the_scheduler(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n[1-2]")
    Profiler(metadata_path=str(tmp_path / "m.json"))
    data = json.loads((tmp_path / "m.json").read_text())
    assert data["scheduler"]["job_id"] == "555"
    assert data["scheduler"]["nodes"] == ["n1", "n2"]


def test_profiler_metadata_scheduler_is_null_outside_a_job(tmp_path, monkeypatch):
    for var in ("SLURM_JOB_ID", "OAR_JOB_ID"):
        monkeypatch.delenv(var, raising=False)
    Profiler(metadata_path=str(tmp_path / "m.json"))
    assert json.loads((tmp_path / "m.json").read_text())["scheduler"] is None
