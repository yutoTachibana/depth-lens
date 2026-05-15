from depth_lens.tasks import ParityTask, get_task


def test_get_task():
    assert isinstance(get_task("parity"), ParityTask)


def test_generate_shape():
    t = ParityTask()
    insts = t.generate(depth=6, n_samples=10, seed=0)
    assert len(insts) == 10
    for x in insts:
        assert x.depth == 6
        toks = x.prompt.split()
        assert toks[-1] == "parity"
        assert len(toks) == 7  # 6 bits + parity
        bits = [int(b) for b in toks[:-1]]
        assert all(b in (0, 1) for b in bits)


def test_target_correct():
    t = ParityTask()
    for d in range(1, 8):
        for inst in t.generate(d, 8, seed=d):
            bits = inst.metadata["bits"]
            assert int(inst.target) == sum(bits) % 2


def test_score_cot_output():
    t = ParityTask()
    inst = t.generate(depth=4, n_samples=1, seed=0)[0]
    # Right answer expressed verbosely
    msg = f"step-by-step: ... final answer: {inst.target}"
    assert t.score(inst, msg) == 1.0
    # Wrong answer
    wrong = "0" if inst.target == "1" else "1"
    assert t.score(inst, f"answer: {wrong}") == 0.0
