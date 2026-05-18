# depth-lens

> **LLM 推論コンピュートの本番運用判断ツール。**
> 3 つの質問、1 つのツール、あなたのワークロードでの実測データ。
>
> [English README](./README.md)

[![tests](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml/badge.svg)](https://github.com/yutoTachibana/depth-lens/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: v2.0](https://img.shields.io/badge/status-v2.0-green.svg)](#ステータス)

LLM を本番に載せているチームは、決まって同じ 3 つの判断に直面します。
たいてい答えは「みんなが使ってるやつ」止まり。depth-lens は、その判断を
**あなたのデータで実測** して、Wilson 95% 信頼区間 + 1 コールあたりのコストつきで、
1 セッション・ランチ代程度で出します。

| 知りたいこと | depth-lens のやり方 | 既存ベンチでの実証 |
|---|---|---|
| **1. どの API tier / thinking budget が最適？** | あなたのプロンプトで全 (model, knob) 組み合わせを sweep し、合格者をコストで並べる | Opus 4.7 → Haiku 4.5 への切替で **年 ~$123k 削減**（10k call/日、同等精度）([finding](docs/findings/v1.0-cost-savings.md)) |
| **2. 自前で open model を運用すべき？それとも API を払い続ける？** | API と vLLM の両方を **1 つの Pareto** に乗せる（$/M-token と $/GPU-hour を同じコスト軸に換算） | K-hop tier 4 (mod-97 K=14) では `gemini-3.1-flash-lite` が 4080 SUPER 上のあらゆる self-hosted を凌駕。tier 1 では self-hosted Llama-3-8B AWQ が **本研究全体で最安の合格 config** ($0.028/1k calls) ([finding](docs/findings/v1.2-self-hosted-vs-api.md)) |
| **3. 推論時のループ計算（looped transformer）に投資する価値はある？** | token CoT API と looped transformer を **同じタスク・同じ精度軸** でプローブ | 訓練 distribution 内では、**925K パラメータの OpenMythos が同じ精度の Claude より ~10,000 倍速い**。外では API が勝つ ([finding](docs/findings/v1.1-architecture-comparison.md)) |

## ヒーロー発見 ── production CI で年 5 桁ドルが浮く

![Opus 4.7 → Haiku 4.5 の切替で 10k call/日のワークロードが年 $123k 削減 — 精度は同じ](docs/findings/figures/hero-cost-savings.png)

**これは実際の depth-lens 出力です。** Anthropic の 4 つの設定が
K-hop tier 4 で全て 1.00 精度を出す中、**コストは ~35 倍も違う**。
「最新・最大を使えば安心」という直感が、静かにこの差額を払い続けています。

> 同じベンチから出てきた他の発見:
> [Claude Haiku 4.5 はハード 2-SAT で default budget だと崩壊、4× budget で復活](docs/findings/v1.0-mini-csp-cross-vendor.md)
> · [2025 年初期の Gemini 2.5 Flash は同世代の Anthropic / OpenAI 廉価推論と比べて唯一弱かった](docs/findings/v1.0-cross-vendor-summary.md#five-structural-findings-depth-lens-surfaced)

## ユースケース 1 ── 「どの API tier？」(本番 cost CI)

**あなた**: Anthropic / OpenAI / Gemini で機能を出している。各ベンダー
3 tier × thinking knob = 9+ 設定。直感は「最新・最大」で、結果として
廉価 tier で十分な精度なのに **20 倍払っている** ケースがある。

**depth-lens が提供するもの**:

- `depth-lens recommend` ── **あなたの JSONL** で全 (model, knob) 組み合わせを probe し、合格者を $/call で並べる単一コマンド
- 各 cell に Wilson 95% 信頼区間 ── 0.95 vs 0.93 がコイン投げかどうか判別可能
- 内蔵 pricing テーブルから $/call を計算、トラフィック量を入れれば $/日・$/年も自動射影
- `--max-latency` で UX SLA を強制 (精度はパスしても遅すぎる config を除外)

**裏付け**: [v1.0 cross-vendor summary](docs/findings/v1.0-cross-vendor-summary.md)
(現行世代 + 2025 世代の全 reasoning モデルを 5 タスクで実測、API 出費合計 ~$14)。
End-to-end の運用シナリオは
[model-downgrade.md](docs/playbook/model-downgrade.md) /
[cost-audit.md](docs/playbook/cost-audit.md) /
[regression-detection.md](docs/playbook/regression-detection.md) の playbook を参照。

## ユースケース 2 ── 「自前で運用すべき？」(build vs buy)

**あなた**: 高コール数で API に月 $X 払っていて、Llama / Qwen / DeepSeek を
GPU 1 枚で運用する選択肢を検討中。本当の問いは抽象的な
「self-hosting で足りるか？」ではなく ──
**「自分のタスク class の上に天井を持つモデルが、SLA 内で $/call 最安になるか？」**

**depth-lens が提供するもの**:

- `vllm:<model>` アダプタ ── OpenAI 互換ローカルサーバーをターゲット。
  compute axis は 2 種類サポート:
  `reasoning_effort` (DeepSeek-R1-Distill, Qwen-Thinking 等の thinking 系) /
  `max_tokens` (Llama-3-8B-Instruct 等の instruct のみのモデル)
- **$/GPU-hour pricing schema** を `cost_per_cell` に追加 ── spec が
  `vllm:*` / `hf:*` / `openmythos` の時、1 コールあたりコストを
  `latency_seconds × $/GPU-hour / 3600` で計算。self-hosted と API が
  **同じコスト軸の同じチャート** に乗る。
- 16 GB 民生 GPU に乗る Docker compose レシピ:
  Llama-3-8B-Instruct AWQ と DeepSeek-R1-Distill-Qwen-1.5B 各 1 ファイル。

**裏付け**: [v1.2 ── APIs vs self-hosted vLLM, 1 つの Pareto](docs/findings/v1.2-self-hosted-vs-api.md)。
ヘッドライン: self-hosted モデルは **正反対の** 精度天井パターンを持つ ──
Llama-3-8B AWQ は tier 1 で勝ち（本研究全体で最安の合格 config）、
tier 4 では **0% 精度**。DeepSeek-R1-Distill-1.5B はその逆。
パラメータ数ではなく **天井位置で選ぶべし**。

![APIs vs self-hosted vLLM, 1 つの Pareto](docs/findings/figures/4way-pareto.png)

参考: [self-hosting-with-vllm.md playbook](docs/playbook/self-hosting-with-vllm.md)。

## ユースケース 3 ── 「推論時ループはスケールするのか？」(アーキ研究)

**あなた**: 推論計算パラダイムを比較したい研究者・応用 ML エンジニア。
2026 年の open question は、**latent-space recursion**
(looped transformer ── OpenMythos, Parcae, Recurrent-Depth) が **token-level CoT**
(extended thinking API) の代替になり得るかどうか。両陣営の marketing 主張はあるが、
**同じチャートで実測した人は誰もいない。**

**depth-lens が提供するもの**:

| パラダイム | depth-lens での実装 | Compute 軸 |
|---|---|---|
| Token-level CoT | `anthropic:*`, `openai:*`, `gemini:*`, `vllm:` thinking | `thinking_budget`, `reasoning_effort`, `thinking_level` |
| Latent-space recursion | `openmythos` (同梱) ── checkpoint が無ければ 7 分で小モデルを訓練 | `n_loops` |

同じ `probe()`、同じ精度軸、同じ Wilson CI。
**両パラダイムを同じ計測器で比べられる唯一の OSS**。

**裏付け**: [v1.1 ── アーキ head-to-head: latent recursion vs token-level CoT](docs/findings/v1.1-architecture-comparison.md)。
OpenMythos の訓練 distribution 内では、**925K パラメータの looped model が
同じ精度の Claude より ~10,000 倍速い**。distribution 外では API が圧倒。
looped-transformer 仮説は支持される ── ただし **訓練深度で bounded**。
depth-lens はその境界を marketing スライドの主張ではなく、
あなたのデータ上で計測可能な事実にします。

参考: [v1.1 OpenMythos 飽和発見](docs/findings/v1.1-cost-vs-latency-per-vendor.md#openmythos-looping-pays-latency-but-the-more-loops--more-depth)
── 「推論時に無限にループ可能」という主張は訓練時の `max_loop_iters` を超えると
**再現しない**。depth-lens が検出した。アーキ側の README は実は予言していた。
今は実データで裏付けられた。

## 30 秒インストール + 最安モデル推薦

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
pip install -e .[anthropic,openai,gemini]

export OPENAI_API_KEY=...     # 必要に応じて ANTHROPIC_API_KEY / GOOGLE_API_KEY も

# あなたの本番プロンプトを JSONL 1 行 1 件で
cat > my_eval.jsonl <<'EOF'
{"prompt": "Compute (47 * 23 + 19) mod 31.", "target": "15", "depth": 1}
{"prompt": "Compute ((11 * 7 - 4) * 3 + 2) mod 41.", "target": "16", "depth": 1}
{"prompt": "Compute (13 * 17 + 8) mod 29.", "target": "26", "depth": 1}
{"prompt": "Compute ((5 * 9 + 7) * 4 - 3) mod 23.", "target": "21", "depth": 1}
{"prompt": "Compute (100 - 7 * 11) mod 19.", "target": "4", "depth": 1}
EOF

# あなたのデータで 95% 精度を満たす最安モデルを探す
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

これだけです。「o4-mini の high effort が本当に必要か、それとも gpt-5-mini の
low で十分か？」という問いに、Wilson 95% CI 付きの実 sweep で根拠を持って
答えられる状態になります。

(`--models` を
`anthropic:claude-haiku-4-5,anthropic:claude-sonnet-4-6,anthropic:claude-opus-4-7`
に差し替えると、ヒーロー plot を生んだ Opus→Haiku 比較が走ります。
同じワークフロー、必要なのは `ANTHROPIC_API_KEY` だけ。)

同じ比較に self-hosted を加えるには:

```bash
# thinking モデル ── OpenAI o-series と同じ reasoning_effort 軸でクリーンに mix
docker compose -f docker/vllm-deepseek-r1-distill.yml up -d

depth-lens recommend \
    --models openai:o4-mini,vllm:deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task custom:my_eval.jsonl:first_int \
    --target-accuracy 0.95 \
    --compute low,medium,high \
    --gpu-hourly-rate 0.50 \
    --n-samples 16 --daily-calls 10000
```

instruct-only モデル (Llama-3-8B-Instruct, Mistral-7B 等、`reasoning_effort` 非対応)
を含める場合は `--compute-axis max_tokens` で vLLM アダプタを応答長キャップ sweep に
切り替え:

```bash
docker compose -f docker/vllm-llama3-8b.yml up -d   # Llama-3-8B-Instruct AWQ をサーブ

depth-lens recommend \
    --models openai:o4-mini,vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --task custom:my_eval.jsonl:first_int \
    --target-accuracy 0.95 \
    --compute-axis max_tokens --compute 256,1024,3000 \
    --gpu-hourly-rate 0.50 \
    --n-samples 16 --daily-calls 10000
```

`--gpu-hourly-rate` で self-hosted のコスト換算レートを指定。
デフォルトは $0.50/GPU-hour (AWS g5 spot 中央値相当)。

## 業務タスクでの検証事例 ── 合成ベンチではなく本番想定で

上記の `recommend` ワークフローを **2 つの本番想定チャットBOTタスク** で実行した結果。
両方とも同じ構造的パターンが出た:
**「念のため高機能モデルを使う」デフォルト選択が strict loss** ──
depth-lens が **約 88% コスト削減** された最適 config を発見、
かつ chat UI として違和感のないレスポンス時間 (1 秒未満) を維持。

### 事例 1 ── 不動産入居者向け 緊急度判定 チャットBOT

賃貸管理会社のチャットBOTが入居者の問い合わせを
`緊急 / 通常 / 翌営業日` に分類。20 件の現実的プロンプト
(水漏れ、ガス漏れ、鍵紛失、契約質問、騒音苦情など)。

| Config | Accuracy | Latency p50 |
|---|---:|---:|
| **`openai:o4-mini @ effort=medium`** ← 採用 | **100%** | **0.52 秒** |
| `openai:gpt-5-mini @ effort=low` | 95% | 0.67 秒 |
| `openai:o4-mini @ effort=high` (「念のため」) | 100% | 0.74 秒 |

- **コスト削減**: `o4-mini @ high` や `gpt-5` をデフォルト選択した場合と比べ **約 88% 削減**
- **depth-lens が surfaced したドメイン洞察**:
  95% config の誤分類 1 件は `通常 → 翌営業日` (安全方向) で、
  `緊急 → 通常` は **ゼロ**。安全要件のあるアプリでは
  accuracy 数字だけでは安価 config の真の安全プロファイルが見えない。

### 事例 2 ── 監視運用会社の 見積もり チャットBOT (MSP)

MSP のチャットBOTが自然文の問い合わせから月額概算を返す
(プラン × 台数 × オプション × ボリューム割引)。
**5 段階の難易度 tier × 53 prompts** で検証。本番風 messy 入力
(typo、敬語/カジュアル混在、暗黙 tier 指定 — 例:
「ミッションクリティカル」→ プレミアム) を含む。

| Config | 全 5 tier accuracy | Latency p50 |
|---|---:|---:|
| **`openai:gpt-5-mini @ effort=low`** ← 採用 | **100% (53/53)** | **0.41 – 0.50 秒** |
| `openai:o4-mini @ effort=medium` | 100% | 0.65 秒 |
| `openai:o4-mini @ effort=high` (「念のため」) | 100% | 0.70 秒 |

- **コスト削減**: 「複雑なロジックは高性能モデルが必要」という直感に逆らって
  実測したら **約 88% 削減**
- **反直感的な発見**: 多段の見積もり計算 + 本番風 messy 入力 でも、
  上位 tier モデルは不要。`gpt-5-mini @ low` が複合プラン、ディスカウント
  条件分岐、口語的日本語 (「がっつり監視で」) を全て 100% で解く。

### 両ケースの共通パターン

1. **「念のため上位モデル」は実測すると strict loss** ──
   精度同じ、コスト高、レイテンシ余裕も生まれない
2. **段階別ベンチ (易 → 本番風) が各 tier の天井を surfacing する** ──
   事例 2 のように「どの候補も天井に達しない」も valid な結果
3. **80-90% のコスト削減は典型** ──
   モデル選定を予断せず、depth-lens で sweep するだけ
4. **本番風入力を day 1 から bench に入れる必要** ──
   合成 tier 1 prompts だけでは上位モデルを過剰推薦してしまう。
   事例 2 の 30 件「本番ログ風」prompts が結論に信頼区間を与えた。

## 全 findings

API キーが取れる全ベンダーで、5 つの組み込みタスク全てに対し、
現行世代 + 1 世代前を fair に並べて測定。さらに self-hosted vLLM と
looped transformer も同じ Pareto に追加。**合計コスト: API ~$14 + ローカル GPU ~30 分。**

### ユースケース 1 ── API ops 向け

| 発見 | 意味 |
|---|---|
| [Opus 4.7 → Haiku 4.5 の切替で 10k call/日タスクで年 ~$123k 削減](docs/findings/v1.0-cost-savings.md) | depth-lens が surfacing する具体的な 4 件の「tier-downgrade」削減を $ で |
| [Cost vs latency: OpenAI gpt-5-mini はトークン単価が安いが o4-mini より 3× 遅い (同精度)](docs/findings/v1.0-cost-savings.md#cost-is-one-axis--latency-is-another) | $/token だけで選ぶと UI レイテンシを焼く。K-hop tier 4 の Pareto frontier は 2 点のみ |
| [Haiku 4.5 はハード 2-SAT で default budget だと崩壊](docs/findings/v1.0-mini-csp-cross-vendor.md) | constraint 系問題に Haiku を使うなら `budget≥4096` 必須。さもなくば 2× エラー率 |
| [Gemini 2.5 Flash は同世代の Anthropic / OpenAI 廉価推論と比べて唯一弱かった](docs/findings/v1.0-cross-vendor-summary.md#five-structural-findings-depth-lens-surfaced) | 2025 世代を 3 ベンダー比較すると Anthropic Sonnet 4 (5月) と o3-mini (1月) は既に天井。Flash だけ崩壊。3.1 Flash-Lite で挽回 |
| [Claude Opus 4.7 は同精度でも (depth × budget) によりコストが 10× ばらつく](docs/findings/v1.0-anthropic-cross-vendor.md) | budget の上限を埋めるのは多くのタスクで strict loss |
| [Per-vendor cost-vs-latency plots (Anthropic / OpenAI / Gemini)](docs/findings/v1.1-cost-vs-latency-per-vendor.md) | ベンダー別 scatter ── Pareto frontier と budget knob の効きが一目 |

### ユースケース 2 ── build vs buy 向け

| 発見 | 意味 |
|---|---|
| [**Self-hosted vLLM (Llama-3-8B / DeepSeek-R1-Distill) vs hosted API ── 1 つの Pareto**](docs/findings/v1.2-self-hosted-vs-api.md) | self-hosted Llama-3-8B AWQ は **tier 1 で本研究全体最安** ($0.028/1k calls)、しかし **tier 4 で 0% 精度**。DeepSeek-R1-Distill-1.5B は tier 4 で 0.75 を出す。tier 4 全体は gemini-3.1-flash-lite が圧勝 ($0.11/1k calls, 1.00 acc)。build-vs-buy を勘ではなくチャートで判断できる |

### ユースケース 3 ── アーキ・パラダイム研究向け

| 発見 | 意味 |
|---|---|
| [**v2.0 ── 3 paradigm を同じ FLOPs 軸で計測する OSS ツール**](docs/findings/v2.0-scaling-law.md) | 真の貢献は **インフラ**: Token-CoT API · Self-hosted vLLM · Looped (OpenMythos 1M/10M/100M) を 1 つの OSS で 5 タスク sweep し、再現可能な probe JSONs を吐く。ヘッドライン (1M-looped が bounded-depth synthetic で gpt-5-mini より ~410,000× FLOPs 効率) は **DL 101 の結果** ── narrow task に train した specialized model が当該タスクで frontier generalist に勝つのは LeNet/MNIST 以来分かってる。**新規性は「あなたのタスクで計測できる tool」であって finding ではない**。Production 判断は v1.0-v1.2 finding 通り「廉価 API tier 使う」が正解 |
| [**OpenMythos (latent recursion) vs Claude (token CoT) head-to-head**](docs/findings/v1.1-architecture-comparison.md) | 訓練 distribution 内では 925K パラメータの looped model が **同精度 Claude より ~10,000× 速い**。外では API が圧倒 |
| [OpenMythos loops-vs-accuracy 飽和](docs/findings/v1.1-cost-vs-latency-per-vendor.md#openmythos-looping-pays-latency-but-the-more-loops--more-depth) | looped transformer の「ループ増やせば深い推論」主張は `training_max_loop_iters` で **飽和**。latency は線形に増えるが精度は伸びない |
| [OpenMythos は K-hop で訓練深度 +1〜2 hops まで外挿](docs/findings/v0.5-openmythos.md) | プロジェクト発端となった seed 実験。同じデータ、同じ軸で |

**[→ v1.0 cross-vendor summary 全文](docs/findings/v1.0-cross-vendor-summary.md)**

## 同梱内容

### 6 つのアダプタファミリー

| 指定 | Compute knob | Cost basis |
|---|---|---|
| `anthropic:<model>` | `thinking_budget_tokens` | API |
| `openai:<model>` | `reasoning_effort` | API |
| `gemini:<model>` | `thinking_budget_tokens` (2.5) / 自動マップ `thinking_level` (3.x) | API |
| `vllm:<model>` | thinking モデルなら `reasoning_effort`、instruct のみなら `max_tokens` (OpenAI 互換ローカルサーバー) | self-hosted ($/GPU-hour) |
| `hf:<hf-model-id>` | `max_thinking_tokens` (CoT 長) | ローカル GPU ($/GPU-hour) |
| `openmythos` | `n_loops` (Recurrent-Depth Transformer) | ローカル GPU ($/GPU-hour) |

API アダプタは thread pool でリクエストを並列化 (`max_concurrent`)。
1000 プロンプトの probe が分単位で終わる。

### 5 つの組み込み probe タスク

| タスク | 深度軸 | 推論の形 |
|---|---|---|
| `k-hop` | K (演算子数) | 順方向合成 (mod-arithmetic) |
| `parity` | n (ビット数) | 集約 (XOR reduction) |
| `graph-reach` | パス長 | 単一 BFS pass |
| `state-tracking` | K (命令数) | ベクトル状態 (2-counter register machine) |
| `mini-csp` | n (変数数) | **探索 / 制約伝播 (2-SAT)** |
| `custom:<jsonl>:<scorer>` | 任意の `depth` フィールド | **自前データを持ち込む** |

`custom:` の組み込みスコアラー: `exact`, `first_int`, `last_int`,
`yes_no`, `contains`, `regex:<pattern>`。冗長な CoT 出力からは
`Final answer: …` 行を自動抽出。

**`llm:<judge-model>:<criterion>` ── LLM-as-judge scorer** で
exact match では測れない open-ended task (要約 / 自由記述 / 多基準) を
判定。Built-in criteria: `correct` / `faithful` / `helpful` /
`concise` / `format` / `polite`。自由 rubric は
`llm:<judge-model>:rubric:<text>`。例:

```bash
depth-lens recommend \
    --models openai:gpt-5-mini,openai:o4-mini \
    --task "custom:./summaries.jsonl:llm:openai:gpt-5-mini:faithful" \
    --target-accuracy 0.85 --n-samples 32
```

判定対象モデルと judge model は **必ず別** にする (self-judging bias 回避)。
Judge cost も実費。2026 年時点で gemini-3.1-flash-lite が最安の competent judge。

### 診断

各 `ProbeResult` が露出するもの:

- `.accuracy` ── `[depth][compute]` グリッド、`[0, 1]` 区間
- `.ci()` ── 全 cell の Wilson 95% 区間
- `.effective_depth(threshold=0.5)` ── どこかの compute level でしきい値を超える最大深度
- `.overthinking(depth, tolerance=0.02)` ── ピーク compute が最大 compute と違うか、その差
- `.cost_per_cell(pricing)` ── $/prediction。token-based pricing
  (`{input, output}` USD-per-1M) と GPU-hour pricing (`{gpu_hourly, gpus}`)
  の両方を受け入れる ── アダプタに合った方を渡せばよい

## CLI

```bash
depth-lens recommend ... # 精度バーを満たす最安モデルを探す (本番ワークフロー)
depth-lens probe ...     # 1 モデルの詳細 sweep
depth-lens compare ...   # 同じタスクで複数モデルを overlay
depth-lens dashboard     # キャッシュ済 probe を Streamlit UI で閲覧
```

各サブコマンドに `--help`。End-to-end 本番シナリオは
[`docs/playbook/`](docs/playbook/) 参照。

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

## やらないこと、他ツールとの比較

depth-lens は意図的に狭い領域に focus しています。
[MMLU](https://github.com/openai/simple-evals) や
[GSM8K](https://github.com/openai/grade-school-math) のような
「フロンティアモデルを 1 つの数字で序列化する」ことは **やりません**。
「モデルが賢いかどうか」もテストしません (本番チームは既にモデルファミリーを選んでいる)。
テストするのは **「そのファミリーのどの設定が、精度バーを満たす最安・最速・最少 GPU
時間か」**。

| | LLMThinkBench | usail-hkust bench | o1 scaling laws | **depth-lens** |
|---|---|---|---|---|
| Compute 軸の曲線（単一点ではなく）| ❌ | partial | ✅ (o1 のみ) | **✅** |
| クロスベンダー (Claude / o-series / Gemini / OSS) | ❌ HF only | partial | ❌ o1 only | **✅** |
| Looped transformer (OpenMythos) | ❌ | ❌ | ❌ | **✅** |
| Self-hosted vLLM を API と同じ軸に乗せる | ❌ | ❌ | ❌ | **✅** |
| 自前 JSONL を持ち込める | ❌ | ❌ | ❌ | **✅** |
| Sweep で $/prediction | ❌ | ❌ | ❌ | **✅** |
| Bounded-depth 合成プローブ | ❌ | partial | ❌ | **✅** |

最も近いアクティブな競合は [LLMThinkBench](https://github.com/ctrl-gaurav/LLMThinkBench)。
HuggingFace モデル限定で固定の operating point の math タスク overthinking を
ターゲットにしており、depth-lens のベンダー API 横断 + compute-axis sweep
とは直交。

## ステータス

- [x] **v0.1 MVP** ── 最初の end-to-end probe (2026 年 5 月)
- [x] **v0.5** ── 4 タスク、5 アダプタ、Wilson CI、キャッシュ、Streamlit ダッシュボード
- [x] **v1.0** ── 6 アダプタファミリー、5 タスク、フル cross-vendor benchmark
  (Anthropic/OpenAI/Gemini、現行 + 2025 前世代)、multi-stage Docker、
  コントリビュータドキュメント、JA 翻訳、GitHub Actions CI (lint + tests)
- [x] **v1.1** ── OpenMythos head-to-head、cross-paradigm Pareto
- [x] **v1.2** ── GPU-hour pricing 付き self-hosted vLLM を同じ Pareto に
- [x] **v2.0** ── 3-paradigm FLOPs-vs-accuracy scaling law (Looped × vLLM × Token-CoT API)、`dict-lookup` タスク、`depth_lens.flops` モジュール、35 cell 完全計測 commit 済
- [ ] **v2.1** ── PyPI 公開、code-generation タスク、追加 adapter (Bedrock / Groq / Together)

ユニットテスト 100+ 件 pass。次の計画は [ROADMAP.md](./ROADMAP.md) を参照。

## インストール variants

```bash
# API のみ (GPU 不要) ── Anthropic, OpenAI, Gemini, dashboard
pip install -e .[anthropic,openai,gemini,dashboard]

# +looped transformer + HuggingFace ローカル probe
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]

# +self-hosted vLLM (vLLM 自体は別途 docker compose で起動する想定)
pip install -e .[anthropic,openai,gemini,dashboard]   # クライアント側は OpenAI SDK だけで十分

# フレームワークのみ (BYO アダプタ)
pip install -e .
```

Python 3.11+。同梱の OpenMythos 訓練ヘルパーは CUDA 前提。
それ以外は CPU でもリモート API 越しでも動作。

## Contributing

Task / Adapter の追加方法 (どちらも ~50 行 + テスト 1 つ) と
同梱実装の規約は [CONTRIBUTING.md](./CONTRIBUTING.md) 参照。

## Citation

```bibtex
@software{depth_lens_2026,
  title  = {depth-lens: Measuring Reasoning Depth Across Model Families},
  author = {yutoTachibana},
  year   = {2026},
  url    = {https://github.com/yutoTachibana/depth-lens}
}
```

## ライセンス

[MIT](./LICENSE).
