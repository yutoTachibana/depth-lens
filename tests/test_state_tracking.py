from depth_lens.tasks import StateTrackingTask, get_task


def test_get_task():
    assert isinstance(get_task("state-tracking"), StateTrackingTask)


def test_generate_shape():
    t = StateTrackingTask()
    insts = t.generate(depth=5, n_samples=10, seed=0)
    assert len(insts) == 10
    for x in insts:
        assert x.depth == 5
        tokens = x.prompt.split()
        # 5 ops + "query" + "1|2"
        assert len(tokens) == 7
        assert tokens[-2] == "query"
        assert tokens[-1] in ("1", "2")
        for op in tokens[:-2]:
            assert op in t.op_names


def test_target_correctness():
    t = StateTrackingTask(modulus=17)
    for d in range(1, 8):
        for inst in t.generate(d, 8, seed=d):
            c1, c2 = inst.metadata["final"]
            q = inst.metadata["query"]
            expected = c1 if q == 1 else c2
            assert int(inst.target) == expected
            assert 0 <= expected < 17


def test_score_extracts_last_int():
    t = StateTrackingTask()
    inst = t.generate(depth=3, n_samples=1, seed=0)[0]
    truth = inst.target
    # Last integer in noisy CoT output
    msg = f"Let's trace: 0 then 1 then 5 ... Final answer: {truth}"
    assert t.score(inst, msg) == 1.0
    # Pure number
    assert t.score(inst, truth) == 1.0
    # Wrong
    wrong = (int(truth) + 1) % t.M
    assert t.score(inst, f"Final answer: {wrong}") == 0.0
