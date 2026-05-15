# depth-lens

> モデル横断で「**推論の深さ**」を計測・可視化するツールキット。

[English README](./README.md)

ループ型 Transformer（OpenMythos, Parcae）、拡張思考API（Claude, o-series, Gemini）、
エージェントループ — 最近の推論系モデルは推論時に「考えるための計算量」を可変に消費します。
MMLU や GSM8K のような既存ベンチマークはその軸を 1 つのスコアに潰してしまいます。
**depth-lens はその軸を可視化します。** 任意の推論システムを入力として、
信頼区間付きの「精度 vs 計算量」曲線、effective depth の推定、overthinking 検出を
モデルファミリーを横断して比較可能な形で出します。

プレアルファ。v0.5 公開、v1.0 進行中。
[ROADMAP.md](./ROADMAP.md) に計画と発見の経緯。

## できること

- **probe エンジン**：タスク深度 × 計算量を掃引、Wilson 95% 信頼区間付きの曲線を出す
- **bounded-depth タスク群**：K-hop モジュラー合成、ビット parity、グラフ到達、2レジスタ状態追跡
- **6 つのアダプタ**：OpenMythos（looped transformer）、HuggingFace 因果 LM、Anthropic Claude 拡張思考、OpenAI o-series 推論努力、Google Gemini 思考モード、vLLM 互換ローカルサーバー
- **自動診断**：`effective_depth`、深度ごとの overthinking 検出、ピーク計算量レポート
- **Streamlit ダッシュボード**：累積した probe 結果をブラウザで掘れる
- **キャッシュ層**：プロットだけ作り直すときに重い probe を再実行しない

## インストール

Python 3.11+ 必須。

```bash
git clone https://github.com/yutoTachibana/depth-lens.git
cd depth-lens
pip install -e .[openmythos,huggingface,anthropic,openai,gemini,dashboard]
```

必要な extras だけ選んで OK。OpenMythos / HuggingFace のローカル推論には CUDA GPU が要りますが、
API アダプタは対応する `*_API_KEY` だけあれば動きます。

## クイックスタート

### 1. 1 モデルを probe する

軽量 OpenMythos を K-hop タスクで学習（民生 GPU で約 7 分）→
ループ数 × タスク深度を掃引:

```bash
depth-lens probe \
    --model openmythos \
    --task k-hop \
    --depths 2,3,4,5,6,7,8,10 \
    --compute 1,2,4,8,16 \
    --train-steps 5000 \
    --save-checkpoint runs/openmythos.pt \
    --plot runs/probe_openmythos.png
```

コンソールに診断が出ます:

```
effective depth (≥0.5 acc at some compute): 7
overthinking @ depth 4: peak=n_loops=4 (acc=1.00)  →  last=n_loops=16 (acc=0.87)
overthinking @ depth 7: peak=n_loops=4 (acc=0.92)  →  last=n_loops=16 (acc=0.45)
```

タスクは `parity` / `graph-reach` / `state-tracking` にも差し替え可能。

### 2. モデル比較

```bash
depth-lens compare \
    --models openmythos,hf:Qwen/Qwen2.5-1.5B-Instruct,anthropic:claude-opus-4-7 \
    --task k-hop \
    --depths 2,4,6,8 \
    --checkpoint runs/openmythos.pt \
    --plot runs/compare.png
```

各アダプタが自分の「計算量ノブ」（n_loops / max_thinking_tokens / thinking_budget /
reasoning_effort）を持ち、それぞれ独立の x 軸で重ねたパネルが深度ごとに並びます。

### 3. インタラクティブ閲覧

```bash
depth-lens dashboard
```

Streamlit がキャッシュ済みの全 probe を読み込み、アダプタ・タスクでフィルター、
信頼区間付き曲線、ヒートマップ、overthinking レポートを掘れます。

## Python API

```python
from depth_lens import probe
from depth_lens.tasks import get_task
from depth_lens.adapters.openmythos_adapter import train_for_task, TrainConfig

task = get_task("parity")
adapter = train_for_task(task, cfg=TrainConfig(steps=2000))

result = probe(adapter, task, depths=[2, 4, 6, 8], n_samples=128)
print(f"effective depth: {result.effective_depth(0.5)}")
print(f"overthinking @ d=8: {result.overthinking(8)}")
print(f"Wilson 95% CIs:\n{result.ci()}")
```

## ビルトインアダプタ

| 指定 | 計算量ノブ | 備考 |
|---|---|---|
| `openmythos` | `n_loops` | チェックポイントが無ければそのタスクで小モデルを訓練 |
| `hf:<hf-model-id>` | `max_thinking_tokens` | 任意の HF 因果 LM を CoT プロンプトで包む。chat template があれば自動利用 |
| `anthropic:<model>` | `thinking_budget_tokens` | Claude の拡張思考。`ANTHROPIC_API_KEY` が要る |
| `openai:<model>` | `reasoning_effort` | o-series の effort（low/medium/high）。`OPENAI_API_KEY` |
| `gemini:<model>` | `thinking_budget_tokens` | Gemini 2.5 の思考モード。`GOOGLE_API_KEY` |
| `vllm:<model>` | `reasoning_effort` | OpenAI 互換ローカルサーバー（vLLM/SGLang/TGI）。`VLLM_BASE_URL` |

## ビルトインタスク

| タスク | 深度軸 | 内容 |
|---|---|---|
| `k-hop` | K（演算子数） | Z/23Z 上のモジュラー合成。加算 2 種＋乗算 2 種で非可換群。Saunshi 2025 系 latent-CoT プローブ |
| `parity` | n（ビット数） | n ビット文字列の XOR。state-tracking の典型 |
| `graph-reach` | パス長 | 小さな DAG 上の yes/no 到達性。正例/負例 50/50 |
| `state-tracking` | K（命令数） | 2 カウンタレジスタ機械（inc1, inc2, swap, add）と最終クエリ。ベクトル状態 |

## 実証された発見

OpenMythos を上記 4 タスクに走らせると、compute-scaling のプロファイルが**質的に異なります**:

- **K-hop**：強い overthinking。訓練深度（loops=4）がピーク、それを超えると劣化
- **Parity**：訓練深度まで単調改善、その後フラット。overthinking なし
- **Graph-reach**：ループを増やしても精度は上がらず ~0.70 でヒューリスティック飽和
- **State-tracking**：overthinking はあるが extrapolation が最良（K=8 でも 0.61）

これらはどれも MMLU のような単一スコアでは見えない事実です。

## ステータス

- [x] **v0.1 MVP** — K-hop、OpenMythos + HF アダプタ、静的プロット、CLI
- [x] **v0.5（進行中）** — +parity / graph-reach / state-tracking、+Anthropic/OpenAI/Gemini/vLLM、Wilson CI、キャッシュ、Streamlit ダッシュボード
- [ ] **v1.0** — cross-vendor benchmark、PyPI 公開、async eval（並列 API 呼び出しは v1.0 で実装済み）

## ライセンス

MIT。[LICENSE](./LICENSE) 参照。

## Citation

```bibtex
@software{depth_lens_2026,
  title  = {depth-lens: Measuring Reasoning Depth Across Model Families},
  author = {yutoTachibana},
  year   = {2026},
  url    = {https://github.com/yutoTachibana/depth-lens}
}
```
