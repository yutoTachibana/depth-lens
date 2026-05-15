# Contributing to depth-lens

Thanks for poking at depth-lens! This document covers the two most common
contribution shapes: **adding a Task** (a new reasoning probe) and **adding
an Adapter** (a new model family).

## Development setup

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
python -m venv .venv && source .venv/bin/activate    # or `uv venv`
pip install -e .[dev,huggingface,openmythos]
pytest                                                 # should be all green
```

A CUDA GPU is required for the bundled OpenMythos training helper, but
nothing else in the codebase assumes CUDA — every adapter is happy on CPU
(or against a remote API).

## Adding a Task

A Task generates `ProbeInstance`s at a controllable `depth` and scores
predictions. Implement it in `depth_lens/tasks/<your_task>.py`:

```python
from depth_lens.tasks.base import ProbeInstance, Task

class MyTask(Task):
    name = "my-task"           # used by `--task my-task` and as registry key
    description = "One-line description."

    def generate(self, depth, n_samples, seed=0):
        rng = random.Random(seed)
        out = []
        for _ in range(n_samples):
            # Build the prompt and the canonical target string.
            # Targets MUST be a single whitespace-delimited token if you want
            # the bundled OpenMythos adapter to learn the task end-to-end.
            prompt = "..."
            target = "..."
            out.append(ProbeInstance(prompt=prompt, target=target,
                                     depth=depth, metadata={...}))
        return out

    def vocab_seed(self):
        # Optional but recommended for tasks where target tokens at K_test > K_train
        # might not appear in low-depth training samples. Return ALL canonical
        # tokens the task will ever emit.
        return [...]

    def score(self, instance, prediction):
        # Default is exact match. Override for lenient scoring (e.g., extract
        # the integer from "Final answer: 7") — see KHopTask.score for the pattern.
        return 1.0 if normalized(prediction) == instance.target else 0.0
```

Then register it in `depth_lens/tasks/__init__.py`:

```python
from depth_lens.tasks.my_task import MyTask

def get_task(name):
    registry = {
        ...,
        MyTask.name: MyTask,
    }
    ...
```

And, if the HF / Anthropic / OpenAI / Gemini adapters should handle it
naturally, add a system-prompt template to each adapter's
`_TASK_INSTRUCTIONS` dict.

### Task conventions

- **Whitespace-separable prompts.** The OpenMythos adapter's naïve
  tokenizer splits on whitespace. Don't bake punctuation into other tokens
  (e.g. write `a -> b` not `a->b`).
- **Lenient scoring.** API models emit verbose chain-of-thought. Your
  scorer should pull the answer out of `Final answer: X` lines or fall
  back to the last integer / last yes-no token.
- **Balanced classes.** For yes/no tasks, generate 50/50. For multiclass,
  generate uniformly over the target space — otherwise a constant
  predictor looks deceptively good.
- **`depth` should mean "compositional depth".** A K=10 instance should
  genuinely require 10 chained inferences, not just have a longer prompt.

### Tests

`tests/test_<task>.py` should at minimum verify:
- Shape / target-range of generated instances
- Deterministic generation given a seed
- Score function correctness on both clean and CoT-formatted predictions

See `tests/test_k_hop.py` for the template.

## Adding an Adapter

A `ModelAdapter` exposes one primitive: `predict(prompts, compute_level)`.
Implement it in `depth_lens/adapters/<your_adapter>.py`:

```python
from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction

class MyAdapter(ModelAdapter):
    name = "my-adapter"

    def __init__(self, model, task_name=None, ...):
        # Load model / open API client / etc.
        ...

    @property
    def compute_axis_name(self):
        # Short label for plots, e.g. "n_loops", "thinking_budget_tokens".
        return "..."

    def default_compute_grid(self):
        # Sensible defaults for `depth-lens probe --model my-adapter` without
        # an explicit --compute flag.
        return [ComputeLevel(value=..., label="...") for ... in ...]

    def predict(self, prompts, compute):
        # Run the model at the given compute level. Return one Prediction
        # per prompt. The .text field is what the task scorer sees, so
        # extract the final answer here if your model emits verbose CoT.
        return [Prediction(text="...", metadata={...}) for ... in ...]

    def teardown(self):
        # Release GPU memory / close client. Default: no-op.
        pass
```

Register it in `depth_lens/adapters/__init__.py::get_adapter` and add a
prefix branch in `depth_lens/cli.py::_build_adapter`.

### Adapter conventions

- **Single primitive.** Adapters should expose ONLY `predict` for the
  probe engine. Anything more elaborate (training, fine-tuning) goes in a
  separate helper module like `openmythos_adapter.train_for_task`.
- **Compute knob is opaque.** depth-lens treats the `value` field of a
  `ComputeLevel` as adapter-specific. Two adapters' compute levels are NOT
  comparable cross-vendor — the compare CLI plots them on per-adapter axes
  for exactly this reason.
- **Strip CoT before returning.** API models emit verbose intermediate
  reasoning. The adapter should pull the final answer out (see
  `_extract_answer` in `hf_adapter.py`) and put it in `Prediction.text`.
  The raw generation goes into `Prediction.metadata["raw_text"]` for
  debugging.
- **Optional dependencies.** Imports of `anthropic`, `openai`,
  `transformers`, etc. happen *inside* `__init__` so the package is
  installable / usable without those SDKs.

### Tests

Adapter tests for API-backed adapters use a mocked SDK injected via
`monkeypatch.setitem(sys.modules, ...)`. See `tests/test_anthropic_adapter.py`
for the pattern. The structure of the test is:

1. Install a fake SDK module that returns a known response.
2. Construct the adapter (it imports the fake SDK).
3. Call `.predict()` and assert the parsed prediction matches.
4. Verify the `--api-key` missing case raises.

## Running tests

```bash
pytest                          # all tests
pytest tests/test_k_hop.py      # one file
pytest -k overthinking          # by keyword
pytest -x --pdb                 # stop on first failure, drop into pdb
```

## Style

- Python 3.11+.
- `ruff` for lint, `black`-compatible formatting (line length 100).
- Type hints on public APIs.
- Docstrings on modules and public classes; one-line comments only when the
  *why* is non-obvious.

## Reporting

- Bugs / feature requests: GitHub issues.
- Empirical findings (a new compute-scaling profile, an interesting
  overthinking case): we'd love a PR adding it as a section in
  `docs/findings/` with the probe JSON committed alongside.

## Code of conduct

Be kind. Assume good faith. Critique the work, not the person.
