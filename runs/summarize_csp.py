import glob
import json
from pathlib import Path

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

PRICES = {
    'anthropic:claude-haiku-4-5':     {'in': 1.0,  'out': 5.0},
    'openai:o4-mini':                 {'in': 1.10, 'out': 4.40},
    'gemini:gemini-2.5-flash':        {'in': 0.30, 'out': 2.50},
    'gemini:gemini-3.1-flash-lite':   {'in': 0.10, 'out': 0.40},
}

paths = sorted(glob.glob(str(ROOT / 'runs/csp_*.json')))
results = []
for p in paths:
    d = json.load(open(p))
    results.append(d)

print(f'{"Model":35s}  {"d=3":>10s} {"d=5":>10s} {"d=7":>10s} {"d=9":>10s}')
print('=' * 80)
for r in results:
    name = r['adapter']
    short = name.replace('anthropic:', '').replace('openai:', '').replace('gemini:', '')
    rows = []
    for di in range(len(r['depths'])):
        cells = r['accuracy'][di]
        rows.append(f'{min(cells):.2f}-{max(cells):.2f}')
    print(f'{short:35s}  {rows[0]:>10s} {rows[1]:>10s} {rows[2]:>10s} {rows[3]:>10s}')

print()
print('=== Per-cell detail (acc at each budget/effort) ===')
for r in results:
    name = r['adapter']
    short = name.replace('anthropic:', '').replace('openai:', '').replace('gemini:', '')
    print(f'\n--- {short} ---')
    print(f'{"depth":>6s}  ' + ' '.join(f'{c["label"][:18]:>18s}' for c in r['compute_grid']))
    for di, d in enumerate(r['depths']):
        cells = r['accuracy'][di]
        print(f'd={d:<5d} ' + ' '.join(f'{c:>18.2f}' for c in cells))

print()
print('=== Cost ===')
total = 0
for r in results:
    name = r['adapter']
    p = PRICES.get(name, {'in': 0, 'out': 0})
    n = r['n_per_cell']
    toks = r.get('tokens_per_cell') or []
    cost = 0
    for row in toks:
        for c in row:
            cost += (c.get('input', 0) * p['in'] + c.get('output', 0) * p['out']) / 1_000_000 * n
    total += cost
    print(f'{name:40s} ${cost:.3f}')
print(f'{"TOTAL":40s} ${total:.3f}')
