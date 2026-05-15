# あなたの推論モデルには「形」がある。標準ベンチマークはそれを潰している。

*depth-lens の紹介 — 推論時計算スケーリングを計測するツールキット*

[English version](./intro.md)

---

最近の推論モデルは奇妙な新しい性質を共有しはじめています：**どれだけ考えさせるかで答えの質が変わる**。
Claude には `extended thinking` のトークン単位の予算がある。OpenAI の o-series には
`reasoning_effort` ノブがある。Gemini は `thinking_budget` を公開している。
オープン重みの looped Transformer（OpenMythos など）は `n_loops` を取る。
エージェント系は諦めるまで繰り返す。

これらは全部、同じ隠れた軸を露出しています：**推論時の計算量**。そして既存ベンチマーク
（MMLU、GSM8K、HumanEval）は全部、その軸を 1 つの数字に平均してしまいます。
ベンチマークスコアからは「計算量を増やすほど単調に良くなるのか」「すぐ頭打ちになるのか」
「予算が多すぎると**逆に悪化する**(overthinking)のか」が読み取れません。

**depth-lens** はこれを解決する小さな OSS です。任意の推論システムに向けて、
タスク深度 × 計算量予算を掃引、信頼区間付きの「精度 vs 計算量」曲線を出し、
overthinking と effective reasoning depth の上限を自動検出します。

```bash
pip install depth-lens[openmythos,huggingface,anthropic,openai,gemini]
depth-lens probe --model anthropic:claude-opus-4-7 --task k-hop --depths 2,4,6,8
```

```
effective depth (≥0.5 acc at some compute): 7
overthinking @ depth 4: peak=think=4096 (acc=1.00)  →  last=think=16384 (acc=0.87)
```

---

## なぜ重要か：3 つのモデル、3 つの異なる「形」

1 日のプロトタイピングセッションで、depth-lens を 4 つの bounded-depth 推論タスクで
試した結果がこれです。同一アーキテクチャ・同一訓練レシピで、925K パラメータの
小さな OpenMythos（Recurrent-Depth Transformer）を 4 タスクそれぞれで訓練して
probe にかけました。compute-scaling 曲線は**質的に異なります**:

| タスク | compute-scaling プロファイル | 意味 |
|---|---|---|
| K-hop モジュラー合成 | **強い overthinking**。訓練深度（loops=4）で精度ピーク、その後劣化。16 ループ時点で K=5 の精度が 1.00 から 0.61 に落ちる。 | モデルは早期に答えを committing してしまい、追加ループで隠れ状態が解から離れていく |
| Parity（n ビット XOR） | **訓練深度まで単調改善**、その後フラット。overthinking なし | ループを全部生産的に使い、それを超えると無害だが効果なし |
| グラフ到達 | **計算量効果ゼロ**。ループ 1, 2, 4, 8, 16 全部 ~0.70。ヒューリスティックで飽和 | モデルは再帰アルゴリズムを学んでおらず、70% のヒューリスティックを見つけて停滞。計算量を増やしても解決しない、別のアプローチが要る |
| 2 カウンタ状態追跡 | **overthinking + 最良の extrapolation**。K-hop と同じく loops=4 がピーク、ただし K=8 でも 0.61 残る | ベクトル状態は丸暗記しにくく、結果として compositional に学習。訓練深度超えで K-hop のような崖落ちにならない |

3 つの挙動はどれも**単一の精度スコアでは見えません**。「OpenMythos は graph-reach で 0.7」という
数字は「ヒューリスティックで飽和」も「計算量を増やせばもっと上がる」も同じに見える。
曲線だけがどちらかを判別します。

---

## depth-lens が実際にやること

中心の抽象は **probe**：タスク深度 × 計算量レベルを掃引し、Wilson 95% 信頼区間付きの
精度グリッドを返す。各アダプタがネイティブの計算量ノブを露出：

- `openmythos` → `n_loops`
- `hf:<model>` → `max_thinking_tokens`（CoT 予算）
- `anthropic:<model>` → `thinking_budget_tokens`
- `openai:<model>` → `reasoning_effort`（low / medium / high）
- `gemini:<model>` → `thinking_budget_tokens`
- `vllm:<model>` → `reasoning_effort`（OpenAI 互換ローカルサーバー）

probe エンジンはどのノブかを気にせず掃引、タスク固有の lenient scorer で予測を採点
（冗長な CoT 出力から "Final answer: X" を抽出）、診断を出す:

- **`effective_depth(threshold)`** — 「ある計算量で」 accuracy 閾値を超える最大のタスク深度
- **`overthinking(depth)`** — ピーク計算量が最大計算量と一致せず、accuracy ドロップが有意のとき検出
- **Wilson 95% CI** — 全セルで計算、曲線にバンドとして描画、`.ci()` でも取得可

キャッシュした probe を **Streamlit ダッシュボード**（`depth-lens dashboard`）で
ブラウズできます。

---

## depth-lens じゃないもの

lm-eval-harness や HELM ではありません。Wikipedia や MATH は同梱しません。
タスクは意図的に小さく、深度可制御で、合成的です。
ポイントは「あなたのモデルが汎用推論で強いか」ではなく
「あなたのモデルが推論に計算量を使うとして、その消費量 vs 品質の曲線はどんな形か」です。

モデル動物園でもありません。再現性ある demo のために小さな OpenMythos 訓練ヘルパーは
同梱していますが、depth-lens は**あなたのモデル**に向けるものです
— フロンティア API でも、OSS の推論モデルでも、研究実装でも。

---

## どこから来たか

depth-lens は [OpenMythos](https://github.com/kyegomez/OpenMythos)
（Anthropic の Claude Mythos アーキテクチャを公開研究から PyTorch で再構築した OSS）
での 1 日実験から生まれました。K-hop 合成タスクで OpenMythos を probe したところ
*ループを増やしても改善しない*ことが分かり、その事実を任意のモデルで露出する
ベンチマーキング基盤が現状無いことに気づきました。

OpenMythos 実験のもう一つの非自明な発見：ACT（Adaptive Computation Time）halting
機構**自体**が外挿性能の bottleneck で、再帰ブロックが原因ではない。
loop=12 の生の隠れ状態には正解が入っていたが、ACT-weighted 出力は loop=4 で
すでに間違いに commit していました。こうした構造的発見を**普通の操作**にすることこそ、
depth-lens の存在意義です。

---

## 次は

- v0.5（今）：5 アダプタファミリー、4 タスク、Wilson CI、キャッシュ、Streamlit ダッシュボード
- v1.0：paper クオリティの cross-vendor benchmark 図（Claude / o-series / Gemini /
  オープン重み推論モデルをフルタスクスイートで）、PyPI 公開、並列 API eval、
  5 番目のタスク（mini-SAT）

「もっと考えさせれば本当に良くなるのか？」と推論モデルを見つめて思ったことがあるなら、
今はコマンド 2 つで答えが出ます。好きなモデルで試してみてください。

→ https://github.com/yutoTachibana/depth-lens
