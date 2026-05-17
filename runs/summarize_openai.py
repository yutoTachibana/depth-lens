import json
from pathlib import Path

_DOCKER_ROOT = Path("/work")
ROOT = _DOCKER_ROOT if _DOCKER_ROOT.exists() else Path(__file__).resolve().parent.parent

# Rough OpenAI pricing in 2026 (USD per 1M tokens)
PRICES = {
  'openai:o4-mini':    {'in': 1.10, 'out': 4.40},
  'openai:gpt-5-mini': {'in': 0.25, 'out': 2.00},
  'openai:gpt-5':      {'in': 1.25, 'out': 10.00},
}

bench = json.load(open(ROOT / 'runs/openai_bench.json'))

print('=== Accuracy across all 3 OpenAI models ===')
print(f'{"":18s} {"tier 1":>10s} {"tier 2":>10s} {"tier 3":>10s} {"tier 4":>10s}')
for r in bench:
    short = r['adapter'].split(':')[-1]
    row = [f'{sum(r["accuracy"][i])/len(r["accuracy"][i]):.2f}' for i in range(len(r['depths']))]
    print(f'{short:18s} ' + ' '.join(f'{x:>10s}' for x in row))

print()
print('=== Token + cost detail per model ===')
for r in bench:
    name = r['adapter']
    short = name.split(':')[-1]
    p = PRICES.get(name, {'in': 0, 'out': 0})
    n = r['n_per_cell']
    toks = r.get('tokens_per_cell')
    if not toks:
        print(f'\n{short}: NO TOKEN DATA (adapter did not record usage)')
        continue
    print(f'\n--- {short} ---')
    efforts = [c['label'].split('=')[-1] for c in r['compute_grid']]
    print(f'{"tier":>6s}  ' + ' '.join(f'effort={e:>6s}' for e in efforts))
    print('-- $/prediction --')
    for di, d in enumerate(r['depths']):
        cells = toks[di]
        costs = [(c.get('input',0)*p['in'] + c.get('output',0)*p['out']) / 1_000_000 for c in cells]
        print(f'tier{d:<3d} ' + ' '.join(f'$ {c:>11.4f}' for c in costs))
    print('-- latency/pred (s) --')
    lat = r.get('latency_per_cell', [])
    for di, d in enumerate(r['depths']):
        cells = lat[di] if di < len(lat) else [0]*len(efforts)
        print(f'tier{d:<3d} ' + ' '.join(f'{c:>13.2f}' for c in cells))

print()
total = 0.0
for r in bench:
    p = PRICES.get(r['adapter'], {'in': 0, 'out': 0})
    n = r['n_per_cell']
    toks = r.get('tokens_per_cell') or []
    sub = 0.0
    for row in toks:
        for c in row:
            sub += (c.get('input',0)*p['in'] + c.get('output',0)*p['out']) / 1_000_000 * n
    total += sub
    print(f'{r["adapter"]:25s} subtotal: ${sub:.3f}')
print(f'{"GRAND TOTAL":25s}           ${total:.3f}')
