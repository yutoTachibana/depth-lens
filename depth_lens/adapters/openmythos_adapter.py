"""
OpenMythos adapter — exposes the n_loops knob as a compute level.

The adapter is a pure inference wrapper: it takes a trained `OpenMythos` model
plus a string→id vocab, tokenizes the prompts on whitespace, runs the forward
pass at the requested loop depth, and returns the argmax token at the position
following "=" as the prediction.

For convenience, a `train_for_task()` helper trains a fresh small model on a
given Task — useful for getting a v0.1 demo working without a pre-existing
checkpoint.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from depth_lens.adapters.base import ComputeLevel, ModelAdapter, Prediction
from depth_lens.tasks.base import ProbeInstance, Task

if TYPE_CHECKING:
    from open_mythos import OpenMythos


PAD_TOKEN = "<pad>"


def _build_vocab(strings: list[str]) -> dict[str, int]:
    """Build a deterministic vocab from a list of whitespace-tokenized strings."""
    seen: dict[str, int] = {PAD_TOKEN: 0}
    for s in strings:
        for tok in s.split():
            if tok not in seen:
                seen[tok] = len(seen)
    return seen


def _tokenize(prompt: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, 0) for tok in prompt.split()]
    if len(ids) > max_len:
        raise ValueError(f"Prompt too long ({len(ids)} > {max_len}): {prompt!r}")
    return ids


def _pad_right(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


@dataclass
class TrainConfig:
    """Knobs for the bundled `train_for_task` helper."""

    steps: int = 6000
    batch_size: int = 256
    depths: tuple[int, ...] = (2, 3, 4, 5, 6)
    lr: float = 3e-4
    warmup: int = 300
    weight_decay: float = 0.01
    dim: int = 128
    n_heads: int = 4
    max_loop_iters: int = 4
    prelude_layers: int = 1
    coda_layers: int = 1
    n_experts: int = 4
    n_shared_experts: int = 1
    n_experts_per_tok: int = 2
    expert_dim: int = 256
    lora_rank: int = 8
    seed: int = 0
    log_every: int = 250


class OpenMythosAdapter(ModelAdapter):
    """Adapter for an OpenMythos model with the n_loops knob as compute axis."""

    name = "openmythos"

    def __init__(
        self,
        model: OpenMythos,
        vocab: dict[str, int],
        max_seq_len: int = 64,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model = model.eval()
        self.vocab = vocab
        self.id_to_tok = {i: t for t, i in vocab.items()}
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if device else next(model.parameters()).device
        if dtype is None:
            dtype = next(model.parameters()).dtype
        self.dtype = dtype

    @property
    def compute_axis_name(self) -> str:
        return "n_loops"

    def default_compute_grid(self) -> list[ComputeLevel]:
        return [ComputeLevel(v, f"loops={v}") for v in (1, 2, 4, 8, 12, 16)]

    @torch.no_grad()
    def predict(self, prompts: list[str], compute: ComputeLevel) -> list[Prediction]:
        ids_lists = [_tokenize(p, self.vocab, self.max_seq_len) for p in prompts]
        ids = _pad_right(ids_lists, pad_id=self.vocab[PAD_TOKEN]).to(self.device)

        autocast = self.device.type == "cuda" and self.dtype != torch.float32
        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self.dtype)
            if autocast
            else _NullCtx()
        )
        with ctx:
            logits = self.model(ids, n_loops=int(compute.value))

        # The prediction we read is the token immediately following the last
        # non-pad token (which the task convention makes "=").
        preds: list[Prediction] = []
        for i, seq in enumerate(ids_lists):
            ans_pos = len(seq) - 1  # in the input, "=" is the last token; logits[ans_pos] predicts next
            tok_id = int(logits[i, ans_pos].argmax().item())
            preds.append(
                Prediction(
                    text=self.id_to_tok.get(tok_id, PAD_TOKEN),
                    metadata={"token_id": tok_id, "n_loops": int(compute.value)},
                )
            )
        return preds

    def teardown(self) -> None:
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Training helper — bundled so v0.1 demo works without a pre-existing ckpt.
# ---------------------------------------------------------------------------


def train_for_task(
    task: Task,
    cfg: TrainConfig | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> OpenMythosAdapter:
    """
    Train a fresh tiny OpenMythos on the given task across `cfg.depths`,
    returning a ready-to-use adapter. Used by the CLI for first-time setup.
    """
    from open_mythos import MythosConfig, OpenMythos

    if cfg is None:
        cfg = TrainConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Start the vocab from the task's declared canonical tokens (if any),
    # then enrich with whatever appears in a sample at each training depth.
    # The seed list ensures targets that only appear at K_test > K_train
    # are in vocab from the start.
    all_strings: list[str] = []
    seed_tokens = task.vocab_seed()
    if seed_tokens:
        all_strings.append(" ".join(seed_tokens))
    for K in cfg.depths:
        for inst in task.generate(K, n_samples=64, seed=cfg.seed + K):
            all_strings.append(inst.prompt + " " + inst.target)
    vocab = _build_vocab(all_strings)
    vocab_size = len(vocab)
    pad_id = vocab[PAD_TOKEN]
    max_len = max(len(s.split()) for s in all_strings) + 2  # +slack

    # Use a generous max_seq_len so test-time depth extrapolation (where K_test
    # can exceed the longest training K) doesn't break the positional buffer.
    mcfg = MythosConfig(
        vocab_size=vocab_size,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_heads,
        max_seq_len=max(128, max_len * 2),
        max_loop_iters=cfg.max_loop_iters,
        prelude_layers=cfg.prelude_layers,
        coda_layers=cfg.coda_layers,
        attn_type="gqa",
        n_experts=cfg.n_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_experts_per_tok=cfg.n_experts_per_tok,
        expert_dim=cfg.expert_dim,
        lora_rank=cfg.lora_rank,
        dropout=0.0,
    )
    model = OpenMythos(mcfg).to(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)

    n_loops = mcfg.max_loop_iters
    model.train()
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        lr_scale = _cosine_with_warmup(step, cfg.steps, cfg.warmup)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * lr_scale

        K = rng.choice(list(cfg.depths))
        insts = task.generate(K, cfg.batch_size, seed=rng.randrange(1 << 30))
        seqs, labels = _encode_for_train(insts, vocab, pad_id)
        seqs = seqs.to(device)
        labels = labels.to(device)

        if device.type == "cuda":
            ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = _NullCtx()
        with ctx:
            logits = model(seqs, n_loops=n_loops)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_id,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if progress and (step % cfg.log_every == 0 or step == 1):
            print(
                f"  [openmythos-train] step {step:5d}/{cfg.steps}  "
                f"K={K}  loss={loss.item():.4f}  elapsed={time.time()-t0:.1f}s"
            )

    return OpenMythosAdapter(model, vocab=vocab, max_seq_len=mcfg.max_seq_len, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_for_train(
    instances: list[ProbeInstance], vocab: dict[str, int], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode (prompt + " " + target) sequences for causal-LM training.
    Loss is taken only at the position predicting the target token (the slot
    after "="). All other positions are masked to pad_id.
    """
    seqs: list[list[int]] = []
    target_positions: list[int] = []
    targets: list[int] = []
    for inst in instances:
        prompt_ids = [vocab[tok] for tok in inst.prompt.split()]
        target_id = vocab[inst.target]
        full = prompt_ids + [target_id]
        seqs.append(full)
        target_positions.append(len(prompt_ids) - 1)  # logits[K+1] predicts target
        targets.append(target_id)

    inputs = _pad_right(seqs, pad_id)
    # Standard LM: inputs[:, :-1] predict inputs[:, 1:]. We score only one slot.
    # Construct labels aligned to inputs[:, :-1].
    seq_in = inputs[:, :-1].contiguous()
    labels = torch.full_like(seq_in, pad_id)
    for i, pos in enumerate(target_positions):
        labels[i, pos] = targets[i]
    return seq_in, labels


def _cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(adapter: OpenMythosAdapter, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "vocab": adapter.vocab,
            "max_seq_len": adapter.max_seq_len,
            "cfg": asdict(adapter.model.cfg),
        },
        path,
    )


def load_checkpoint(
    path: Path, device: str | torch.device | None = None
) -> OpenMythosAdapter:
    from open_mythos import MythosConfig, OpenMythos

    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = ckpt["cfg"]
    cfg = MythosConfig(**{k: v for k, v in saved_cfg.items() if k in MythosConfig.__dataclass_fields__})
    model = OpenMythos(cfg).to(device=device)
    model.load_state_dict(ckpt["state_dict"])
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return OpenMythosAdapter(
        model, vocab=ckpt["vocab"], max_seq_len=ckpt["max_seq_len"], device=device, dtype=dtype
    )
