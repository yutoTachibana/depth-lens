import json
from pathlib import Path

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

PRICES = {
  'anthropic:claude-haiku-4-5':  {'in': 1.0,  'out': 5.0,  'thinking': 5.0},
  'anthropic:claude-sonnet-4-6': {'in': 3.0,  'out': 15.0, 'thinking': 15.0},
  'anthropic:claude-opus-4-7':   {'in': 15.0, 'out': 75.0, 'thinking': 75.0},
}

bench = json.load(open(ROOT / 'runs/bench.json'))
opus_cache = json.load(open(ROOT / '.cache/depth-lens/probes/thinking.json'))

print('=== Accuracy across all 3 models ===')
print(f'{"":18s} {"tier 1":>8s} {"tier 2":>8s} {"tier 3":>8s} {"tier 4":>8s}')
for r in bench:
    short = r['adapter'].split(':')[-1].replace('claude-', '')
    row = [f'{sum(r["accuracy"][i])/len(r["accuracy"][i]):.2f}' for i in range(len(r['depths']))]
    print(f'{short:18s} ' + ' '.join(f'{x:>8s}' for x in row))

print()
print('=== Opus 4.7 detail (full token/latency data — survived) ===')
budgets = [c['label'].split('=')[-1] for c in opus_cache['compute_grid']]
print(f'\n{"depth":>6s}  ' + ' '.join(f'budget={b:>6s}' for b in budgets))
print('-- acc --')
for di, d in enumerate(opus_cache['depths']):
    cells = opus_cache['accuracy'][di]
    print(f'tier{d}  ' + ' '.join(f'{c:>13.2f}' for c in cells))
print('-- $/prediction --')
p = PRICES['anthropic:claude-opus-4-7']
for di, d in enumerate(opus_cache['depths']):
    cells = opus_cache['tokens_per_cell'][di]
    costs = [(c.get('input',0)*p['in'] + c.get('output',0)*p['out']) / 1_000_000 for c in cells]
    print(f'tier{d}  ' + ' '.join(f'$ {c:>11.4f}' for c in costs))
print('-- latency/pred --')
for di, d in enumerate(opus_cache['depths']):
    cells = opus_cache['latency_per_cell'][di]
    print(f'tier{d}  ' + ' '.join(f'{c:>12.2f}s' for c in cells))

print()
total = 0.0
for di in range(len(opus_cache['depths'])):
    for c in opus_cache['tokens_per_cell'][di]:
        total += (c.get('input',0)*p['in'] + c.get('output',0)*p['out']) / 1_000_000 * opus_cache['n_per_cell']
print(f'Opus probe spend (alone): ${total:.3f}')
print(f'Estimated 3-model total this run: ${total * 1.4:.2f} (Opus is the dominant cost)')
