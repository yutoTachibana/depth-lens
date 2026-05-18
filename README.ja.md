# depth-lens

> **あなたのデータで「精度バーを満たす最安 LLM 設定」を選ぶ ── 他人の benchmark ではなく。**
>
> 1 つの CLI で全 (model, knob) 組み合わせを sweep。Wilson 95% 信頼区間、1 コール単価、p50 レイテンシ、ベンダー横断。**1 監査あたり ~$1 / 10 分**。
>
> [English README](./README.md)

[![tests](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml/badge.svg)](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: v2.1](https://img.shields.io/badge/status-v2.1-green.svg)](#ステータス)

![Opus 4.7 → Haiku 4.5 で 10k call/日のワークロードが年 $123k 削減 — 精度は同じ](docs/findings/figures/hero-cost-savings.png)

**上記は実際の `depth-lens recommend` 出力です。** Anthropic の 4 つの config が K-hop tier 4 の prompt 集合で **全て 1.00 精度** を達成しつつ、**コストは ~35 倍違う**。「最新・最大を使えば安心」という直感が静かにこの差額を払い続けています。depth-lens は **あなたの** prompt で最安合格 tier を 10 分以内に見つけます。30 秒インストールはすぐ下。

## 30 秒: インストール、実行、決定

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
pip install -e .[anthropic,openai,gemini]

export OPENAI_API_KEY=...     # 必要に応じて ANTHROPIC_API_KEY / GOOGLE_API_KEY も

# あなたの本番 prompt を JSONL に
cat > my_eval.jsonl <<'EOF'
{"prompt": "Compute (47 * 23 + 19) mod 31.", "target": "15", "depth": 1}
{"prompt": "Compute ((11 * 7 - 4) * 3 + 2) mod 41.", "target": "16", "depth": 1}
{"prompt": "Compute (13 * 17 + 8) mod 29.", "target": "26", "depth": 1}
{"prompt": "Compute ((5 * 9 + 7) * 4 - 3) mod 23.", "target": "21", "depth": 1}
{"prompt": "Compute (100 - 7 * 11) mod 19.", "target": "4", "depth": 1}
EOF

depth-lens recommend \
    --models openai:gpt-5-mini,openai:o4-mini \
    --task custom:my_eval.jsonl:first_int \
    --target-accuracy 0.95 \
    --max-latency 3.0 \
    --n-samples 16 \
    --daily-calls 10000
```

```
============================================================================================
Target accuracy ≥ 0.95  ·  Max latency ≤ 3.00s/pred
Probed 6 configurations, 6 passing.
============================================================================================

✅ Passing (cheapest first):
  openai:gpt-5-mini     d=1  effort=low      acc=1.00   $0.354/k-pred   0.45s/pred  ← cheapest
  openai:gpt-5-mini     d=1  effort=medium   acc=1.00   $0.485/k-pred   0.59s/pred
  openai:o4-mini        d=1  effort=low      acc=1.00   $0.736/k-pred   0.29s/pred  ← fastest
  openai:gpt-5-mini     d=1  effort=high     acc=1.00   $0.886/k-pred   0.69s/pred
  openai:o4-mini        d=1  effort=medium   acc=1.00   $1.061/k-pred   0.37s/pred
  openai:o4-mini        d=1  effort=high     acc=1.00   $1.365/k-pred   0.40s/pred

⚡ Cost-vs-speed tradeoff among passing configs:
  Cheapest is 1.5× slower than fastest; fastest costs 2.1× more per call.

============================================================================================
At 10,000 calls/day with the cheapest passing config:
  openai:gpt-5-mini @ effort=low
  → $3.54/day  $1,291/year

  Switching from openai:o4-mini @ effort=high ($13.65/day)
  saves $10.11/day = $3,691/year (74% reduction)
```

「上位 config / 高 effort が本当に必要か?」 ── **あなたの** prompt 上で Wilson 95% CI 付きの実測 sweep で答えが出る。`--models` を Anthropic / Gemini / vLLM (self-hosted) に差し替えて同じワークフロー。

## エビデンス: 3 つの本番想定タスクで 3 つの測定

3 つの異なるスコアリングニーズに対応する production-style chatbot タスクで end-to-end 実行。同じ `recommend` ワークフローで 3 種類の scorer を活用。

### Case 1 ── 不動産入居者問い合わせ 緊急度判定

入居者の問い合わせを `緊急 / 通常 / 翌営業日` に分類。20 件の現実的 prompt: 水漏れ / ガス漏れ / 鍵紛失 / 契約質問 / 騒音苦情。

| Config | Accuracy | Latency p50 |
|---|---:|---:|
| **`openai:o4-mini @ effort=medium`** ← 採用 | **100%** | **0.52 秒** |
| `openai:gpt-5-mini @ effort=low` | 95% | 0.67 秒 |
| `openai:o4-mini @ effort=high` (「念のため」) | 100% | 0.74 秒 |

**コスト削減: ~88%** vs `o4-mini @ high` や `gpt-5` のデフォルト選択。
**depth-lens が surfacing したドメイン洞察**: 95% config の唯一の誤分類は `通常 → 翌営業日` (安全方向)。`緊急 → 通常` のエラーゼロ ── accuracy 数字だけでは安価 config の真の安全プロファイルが見えない。

### Case 2 ── システム監視会社 見積もり チャットBOT (MSP)

自然文の問い合わせから月額概算を計算 (プラン × 台数 × オプション × ボリューム割引)。**5 難易度 tier × 53 prompts** ── typo、敬語/カジュアル混在、暗黙 tier 指定 ("ミッションクリティカル" → premium) を含む。

| Config | 全 5 tier acc | Latency p50 |
|---|---:|---:|
| **`openai:gpt-5-mini @ effort=low`** ← 採用 | **100% (53/53)** | **0.41 – 0.50 秒** |
| `openai:o4-mini @ effort=medium` | 100% | 0.65 秒 |
| `openai:o4-mini @ effort=high` (「念のため」) | 100% | 0.70 秒 |

**コスト削減: ~88%** vs「複雑なロジックは高性能モデルが必要」直感に逆らって実測した結果。
**反直感的な発見**: 多段の見積もり計算 + 本番風 messy 入力 でも、上位 tier モデルは不要。`gpt-5-mini @ low` が複合プラン、ディスカウント条件分岐、口語的日本語 (「がっつり監視で」) を全て 100% で解く。

### Case 3 ── 入居者返信品質、LLM による判定 (v2.1)

同じ物件管理 BOT だが、**自由記述の返信生成**。品質は別の LLM が 3 基準ルブリック (敬語使用 / 具体的な問題に対応 / 明確な次のアクション提示) で判定。12 件の問い合わせ (緊急 / 事務手続き / 規約 / 苦情 / 設備故障)。

| Config | 3 基準すべて合格 | 1 返信あたり latency |
|---|---:|---:|
| **`openai:gpt-5-mini @ effort=low`** ← 採用 | **100% (12/12)** | **1.4 秒** |
| `openai:o4-mini @ effort=high` | 100% (12/12) | 1.8 秒 |
| `openai:gpt-5-mini @ effort=high` (「念のため」) | 75% (9/12) | **15.6 秒** ← 実用不可 |

**反直感**: `gpt-5-mini` は effort 上げると acc が **下がる** (low 100% → high 75%) ── 冗長返信が「具体的な問題に対応 / 次のアクション」基準を満たさなくなる。`o4-mini` は逆方向の曲線。**最適 effort は (model × task) ごと、汎用ルールなし**。
**なぜこの case が重要か**: v2.1 まで depth-lens は自由記述の返信を測れなかった ── case 1・2 のような structured タスクのみ。新 [`llm:` scorer](#自分のタスクは測れるか-3-つの-scorer-family) がこれを可能にした。

[詳細 case study →](docs/findings/v2.1-llm-judge-case-study.md)

### 3 ケース共通の 5 パターン

1. **「念のため上位モデル」は実測すると strict loss** ── 精度同じ、コスト高、レイテンシ余裕も生まれない。Case 3 はこれを更に強める ── 自由記述では effort 上げると acc が **下がる** ことすらある
2. **段階別ベンチ (易 → 本番風) が各 tier の天井を surfacing する** ── 事例 2 のように「どの候補も天井に達しない」も valid な結果
3. **80-90% のコスト削減は典型** ── モデル選定を予断せず、depth-lens で sweep するだけ
4. **本番風入力を day 1 から bench に入れる必要** ── 合成 tier 1 prompts だけでは上位モデルを過剰推薦してしまう。事例 2 の 30 件「本番ログ風」prompts が結論に信頼区間を与えた
5. **モデル選定より scorer 選定が重要** ── 3 ケースで scorer 3 種類 (structured 分類 = `exact`、数値抽出 = `first_int`、open-ended LLM 判定 = `llm:`) すべてカバー。新規 production task はこの 3 つのどれかに収まる

## 自分のタスクは測れるか? 3 つの scorer family

| Family | Spec 形式 | 用途 |
|---|---|---|
| **Structured** | `exact`, `first_int`, `last_int`, `yes_no`, `contains` | 分類、数値答え、yes/no 判定。上記 Case 1, 2 |
| **Regex** | `regex:<pattern>` | フォーマット検査、「答えがこの形に合致するか」 |
| **LLM-as-judge** | `llm:<judge-model>:<criterion>` または `llm:<judge-model>:rubric:<text>` | Open-ended 出力: 要約、自由記述 Q&A、多基準判定。上記 Case 3 |

`llm:` の組み込み criterion: `correct` / `faithful` / `helpful` / `concise` / `format` / `polite`。自由 rubric は `:rubric:` の後に任意テキスト。

```bash
# LLM-judge 例: gpt-5-mini で要約 faithfulness を判定
depth-lens recommend \
    --models openai:gpt-5-mini,openai:o4-mini \
    --task "custom:./summaries.jsonl:llm:openai:gpt-5-mini:faithful" \
    --target-accuracy 0.85 --n-samples 32
```

判定対象モデルとは必ず違う (理想的にはより安い) judge を選び self-judging bias 回避。2026 年時点では gemini-3.1-flash-lite が最安の competent judge。

この 3 family で **production AI タスクのほぼ全てが測れる** ── 分類 / structured extraction / RAG-faithfulness / 顧客対応返信品質 / コードレビュー / トーン検査。合わないタスクがあれば issue 立ててください。

## 同梱内容

### 6 アダプタファミリー

| Spec | Compute knob | Cost basis |
|---|---|---|
| `anthropic:<model>` | `thinking_budget_tokens` | API ($/M-token) |
| `openai:<model>` | `reasoning_effort` | API ($/M-token) |
| `gemini:<model>` | `thinking_budget_tokens` (2.5) / 自動マップ `thinking_level` (3.x) | API ($/M-token) |
| `vllm:<model>` | thinking モデルなら `reasoning_effort`、instruct のみなら `max_tokens` (OpenAI 互換ローカルサーバー) | self-hosted ($/GPU-hour) |
| `hf:<hf-model-id>` | `max_thinking_tokens` (CoT 長) | local GPU ($/GPU-hour) |
| `openmythos` | `n_loops` (Recurrent-Depth Transformer) | local GPU ($/GPU-hour) |

API アダプタは thread pool でリクエスト並列化 (`max_concurrent`)。1000 prompt の probe が分単位で終わる。

### 5 + 1 組み込み probe タスク

| Task | Depth axis | 推論の形 |
|---|---|---|
| `k-hop` | K (演算子数) | 順方向合成 (mod-arithmetic) |
| `parity` | n (ビット数) | 集約 (XOR reduction) |
| `graph-reach` | パス長 | 単一 BFS pass |
| `state-tracking` | K (命令数) | 2-counter register machine |
| `mini-csp` | n (変数数) | **探索 / 制約伝播 (2-SAT)** |
| `dict-lookup` | n (ペア数) | **構造化入力からのフィールド抽出** (v2.0) |
| `custom:<jsonl>:<scorer>` | 任意の `depth` フィールド | **自前データを持ち込む** |

### 各 `ProbeResult` が露出する診断

- `.accuracy` ── `[depth][compute]` グリッド、`[0, 1]` 区間
- `.ci()` ── 全 cell の Wilson 95% 区間
- `.effective_depth(threshold=0.5)` ── どこかの compute level でしきい値を超える最大深度
- `.overthinking(depth, tolerance=0.02)` ── ピーク compute が最大 compute と違うか、その差
- `.cost_per_cell(pricing)` ── $/prediction。token-based (`{input, output}` USD-per-1M) または GPU-hour (`{gpu_hourly, gpus}`) ── アダプタに合った方を渡す

## CLI

```bash
depth-lens recommend ... # 精度バーを満たす最安モデルを探す (本番ワークフロー)
depth-lens probe ...     # 1 モデルの詳細 sweep
depth-lens compare ...   # 同じタスクで複数モデルを overlay
depth-lens dashboard     # キャッシュ済 probe を Streamlit UI で閲覧
```

各サブコマンドに `--help`。End-to-end 本番シナリオは [`docs/playbook/`](docs/playbook/) を参照:
[model-downgrade](docs/playbook/model-downgrade.md) · [cost-audit](docs/playbook/cost-audit.md) · [regression-detection](docs/playbook/regression-detection.md) · [self-hosting-with-vllm](docs/playbook/self-hosting-with-vllm.md)。

## Python API

```python
from depth_lens import probe
from depth_lens.tasks import get_task
from depth_lens.adapters.anthropic_adapter import AnthropicAdapter

task = get_task("mini-csp")
adapter = AnthropicAdapter(model="claude-haiku-4-5", task_name="mini-csp")
result = probe(adapter, task, depths=[3, 5, 7, 9], n_samples=16)

print(f"effective depth: {result.effective_depth(0.5)}")
print(f"overthinking @ d=9: {result.overthinking(9)}")
print(f"$/pred @ d=9 mid budget: {result.cost_per_cell({'input': 1.0, 'output': 5.0})[3, 1]}")
```

## depth-lens の "やらないこと"

- [MMLU](https://github.com/openai/simple-evals) や [GSM8K](https://github.com/openai/grade-school-math) のような **leaderboard はやらない**。あれらは canonical benchmark でフロンティアモデルを序列化するもの。production チームは既にモデルファミリーを選んでおり、その family 内で tune したい
- 「モデルが賢いか」もテストしない。テストするのは **「そのファミリーのどの設定が、あなたの精度バーを満たす最安・最速・最少 GPU 時間か」**
- ホスト型ダッシュボードはやらない。OSS は JSON と plot をローカル生成。これに乗っかるホスト型サービスは scope 外

| 機能 | LLMThinkBench | usail-hkust bench | o1 scaling laws | **depth-lens** |
|---|---|---|---|---|
| Compute 軸の曲線 (単一点ではなく) | ❌ | partial | ✅ (o1 のみ) | **✅** |
| クロスベンダー (Claude / o-series / Gemini / OSS) | ❌ HF only | partial | ❌ o1 only | **✅** |
| Self-hosted vLLM を API と同じ軸に乗せる | ❌ | ❌ | ❌ | **✅** |
| Looped transformer (OpenMythos) | ❌ | ❌ | ❌ | **✅** |
| 自前 JSONL を持ち込める | ❌ | ❌ | ❌ | **✅** |
| **Open-ended task の LLM-as-judge scorer** | ❌ | ❌ | ❌ | **✅** |
| Sweep で $/prediction | ❌ | ❌ | ❌ | **✅** |

最も近いアクティブな競合は [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench) ── HF モデル限定で固定 operating point の math タスク overthinking が target。depth-lens のベンダー API 横断 + compute-axis sweep とは直交。

## どんなユースケースで使うか

| 知りたいこと | `depth-lens recommend` が出すもの | ヘッドラインエビデンス |
|---|---|---|
| **1. どの API tier / thinking budget が最適？** | あなたの prompt で全 (model, knob) の最安合格 config | Opus 4.7 → Haiku 4.5 で 10k call/日タスクで **年 ~$123k 削減**、同等精度 ([finding](docs/findings/v1.0-cost-savings.md)) |
| **2. 自前で open model を運用すべき？それとも API を払い続ける？** | API と vLLM を 1 つの Pareto に ($/M-token と $/GPU-hour を同じコスト軸に) | K-hop tier 4 では `gemini-3.1-flash-lite` が 4080 SUPER 上のあらゆる self-hosted を凌駕。tier 1 では Llama-3-8B AWQ が **最安** ($0.028/1k calls) ([finding](docs/findings/v1.2-self-hosted-vs-api.md)) |
| **3. 自由記述の出力品質も測れる？** | LLM-judge スコアを structured scorer と同じ Wilson CI で | Case 3 ── `gpt-5-mini @ low` が 3 基準 rubric で勝つ。effort 上げると acc 下がる ([finding](docs/findings/v2.1-llm-judge-case-study.md)) |

研究用途 (paradigm scaling、推論時計算量の計測インフラ) には [v2.0 cross-paradigm 計測 plot](docs/findings/v2.0-scaling-law.md) を参照 ── Token-CoT API · Self-hosted vLLM · Looped transformer を同じ FLOPs 軸に乗せるツール。あくまで **計測ツール** が contribution であり、その背後にある観察 (特化モデルが汎用に当該タスクで勝つ) はディープラーニング教科書的知識。

## 全 findings

API キーが取れる全ベンダーで、組み込みタスク全てに対し、現行世代 + 1 世代前を fair に並べて測定。合計コスト: **API ~$14 + ローカル GPU ~30 分 + LLM-judge ~$1** (Case 3)。

| ユースケース | 発見 | 意味 |
|---|---|---|
| API ops | [Opus 4.7 → Haiku 4.5 で 10k call/日タスクで年 ~$123k 削減](docs/findings/v1.0-cost-savings.md) | depth-lens が surfacing する具体的な 4 件の tier-downgrade 削減 を $ で |
| API ops | [gpt-5-mini はトークン単価が安いが o4-mini より 3× 遅い](docs/findings/v1.0-cost-savings.md#cost-is-one-axis--latency-is-another) | $/token だけで選ぶと UI レイテンシを焼く |
| API ops | [Haiku 4.5 はハード 2-SAT で default budget だと崩壊](docs/findings/v1.0-mini-csp-cross-vendor.md) | constraint 系問題に Haiku を使うなら `budget≥4096` 必須 |
| API ops | [Gemini 2.5 Flash は同世代 Anthropic / OpenAI 廉価推論と比べて唯一弱かった](docs/findings/v1.0-cross-vendor-summary.md#five-structural-findings-depth-lens-surfaced) | 3.1 Flash-Lite で挽回 |
| API ops | [Claude Opus 4.7 は同精度でも (depth × budget) によりコストが 10× ばらつく](docs/findings/v1.0-anthropic-cross-vendor.md) | budget の上限を埋めるのは多くのタスクで strict loss |
| API ops | [ベンダー別 cost-vs-latency plot](docs/findings/v1.1-cost-vs-latency-per-vendor.md) | ベンダー別 scatter ── Pareto frontier と budget knob の効きが一目 |
| Build vs buy | [Self-hosted vLLM vs hosted API を 1 つの Pareto に](docs/findings/v1.2-self-hosted-vs-api.md) | Llama-3-8B AWQ は **tier 1 で最安**、tier 4 で **0% 精度**。DeepSeek-R1-Distill-1.5B は tier 4 で 0.75。Build-vs-buy を勘ではなくチャートで判断 |
| Open-ended | [顧客返信品質を LLM-as-judge で (Case 3)](docs/findings/v2.1-llm-judge-case-study.md) | 自由記述タスクで `gpt-5-mini` は effort 上げると acc 下がる。最適 effort は (model × task) ごと |
| 研究 / ツール | [v2.0 ── 3 paradigm を同じ FLOPs 軸で](docs/findings/v2.0-scaling-law.md) | Token-CoT API · Self-hosted vLLM · Looped (OpenMythos 1M/10M/100M) を 1 つの軸に乗せるインフラ。24,000-410,000× FLOPs 比はディープラーニング教科書的結果、**ツール** が contribution |
| 研究 | [OpenMythos vs Claude head-to-head](docs/findings/v1.1-architecture-comparison.md) | 訓練 distribution 内では 925K param looped が Claude より **~10,000× 速い**。外では API が圧倒 |
| 研究 | [OpenMythos loops-vs-accuracy 飽和](docs/findings/v1.1-cost-vs-latency-per-vendor.md#openmythos-looping-pays-latency-but-the-more-loops--more-depth) | 「ループ増やせば深い推論」主張は `training_max_loop_iters` で飽和 |
| 研究 | [OpenMythos は K-hop で訓練深度 +1〜2 hops まで外挿](docs/findings/v0.5-openmythos.md) | プロジェクト発端となった seed 実験 |

**[→ v1.0 cross-vendor summary 全文](docs/findings/v1.0-cross-vendor-summary.md)**

## ステータス

- [x] **v0.1 MVP** ── 最初の end-to-end probe (2026 年 5 月)
- [x] **v0.5** ── 4 タスク、5 アダプタ、Wilson CI、キャッシュ、Streamlit ダッシュボード
- [x] **v1.0** ── 6 アダプタファミリー、5 タスク、フル cross-vendor benchmark (Anthropic/OpenAI/Gemini、現行 + 2025 前世代)、multi-stage Docker、GitHub Actions CI
- [x] **v1.1** ── OpenMythos head-to-head、cross-paradigm Pareto
- [x] **v1.2** ── GPU-hour pricing 付き self-hosted vLLM を同じ Pareto に
- [x] **v2.0** ── 3-paradigm FLOPs 計測ツール、`dict-lookup` タスク、`depth_lens.flops` モジュール
- [x] **v2.1** ── Open-ended task 向け LLM-as-judge scorer (`llm:<judge>:<criterion>`)、入居者返信 case study
- [ ] **v2.2** ── PyPI 公開、judge cost を `recommend` の $/k-pred に統合、`--free-form` CLI flag、code-generation タスク

ユニットテスト 128 件 pass。次の計画は [ROADMAP.md](./ROADMAP.md) を参照。

## インストール variants

```bash
# API のみ (GPU 不要) ── Anthropic, OpenAI, Gemini, dashboard
pip install -e .[anthropic,openai,gemini,dashboard]

# +looped transformer + HuggingFace ローカル probe
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]

# +self-hosted vLLM (vLLM は別途 docker compose で起動する想定)
pip install -e .[anthropic,openai,gemini,dashboard]   # クライアント側は OpenAI SDK だけで十分

# フレームワークのみ (BYO アダプタ)
pip install -e .
```

Python 3.11+。同梱の OpenMythos 訓練ヘルパーは CUDA 前提。それ以外は CPU でもリモート API 越しでも動作。

## Contributing

Task / Adapter の追加方法 (どちらも ~50 行 + テスト 1 つ) と同梱実装の規約は [CONTRIBUTING.md](./CONTRIBUTING.md) 参照。

## Citation

```bibtex
@software{depth_lens_2026,
  title  = {depth-lens: Measuring Inference-Time Compute for LLM Production Decisions},
  author = {yutoTachibana},
  year   = {2026},
  url    = {https://github.com/yutoTachibana/depth-lens}
}
```

## ライセンス

[MIT](./LICENSE).
