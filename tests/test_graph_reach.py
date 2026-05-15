from depth_lens.tasks import GraphReachabilityTask, get_task


def test_get_task():
    assert isinstance(get_task("graph-reach"), GraphReachabilityTask)


def test_generate_yes_no_balance():
    t = GraphReachabilityTask()
    insts = t.generate(depth=3, n_samples=20, seed=0)
    yeses = sum(1 for x in insts if x.target == "yes")
    nos = sum(1 for x in insts if x.target == "no")
    assert yeses == nos == 10


def test_prompt_is_whitespace_tokenizable():
    """The prompt must split cleanly into single-character / short tokens so
    naive whitespace tokenizers build a small finite vocab."""
    t = GraphReachabilityTask()
    insts = t.generate(depth=4, n_samples=5, seed=0)
    allowed_specials = {"edges", ":", ";", "reach", "->", "?"}
    for inst in insts:
        for tok in inst.prompt.split():
            if tok in allowed_specials:
                continue
            # Otherwise must be a node name (single ascii letter)
            assert len(tok) == 1 and tok.isalpha(), f"Unexpected token: {tok!r} in {inst.prompt!r}"


def test_unique_path_exists_for_yes():
    """For positive instances, the chain should actually connect start → goal."""
    t = GraphReachabilityTask()
    insts = t.generate(depth=4, n_samples=10, seed=1)
    for inst in insts:
        if inst.target != "yes":
            continue
        edges = set(inst.metadata["edges"])
        s = inst.metadata["start"]
        g = inst.metadata["goal"]
        # BFS
        seen = {s}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for a, b in edges:
                    if a == u and b not in seen:
                        seen.add(b)
                        nxt.append(b)
            frontier = nxt
        assert g in seen, f"yes-instance must have path: {inst.prompt}"


def test_goal_not_in_path_for_no():
    """For negative instances, goal should NOT be reachable from start."""
    t = GraphReachabilityTask()
    insts = t.generate(depth=4, n_samples=10, seed=2)
    for inst in insts:
        if inst.target != "no":
            continue
        edges = set(inst.metadata["edges"])
        s = inst.metadata["start"]
        g = inst.metadata["goal"]
        seen = {s}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for a, b in edges:
                    if a == u and b not in seen:
                        seen.add(b)
                        nxt.append(b)
            frontier = nxt
        assert g not in seen, f"no-instance must NOT have path: {inst.prompt}"


def test_score_cot_output():
    t = GraphReachabilityTask()
    inst = t.generate(depth=3, n_samples=1, seed=0)[0]
    msg = f"Let's trace... Final answer: {inst.target}"
    assert t.score(inst, msg) == 1.0
    wrong = "no" if inst.target == "yes" else "yes"
    assert t.score(inst, f"Answer: {wrong}") == 0.0
    # True/false synonyms
    syn = "true" if inst.target == "yes" else "false"
    assert t.score(inst, f"Answer: {syn}") == 1.0
