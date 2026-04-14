# インテグレーションガイド

> 本ドキュメントは、皆さんのVLA/ポリシーモデルを AIRoA 評価パイプラインに適応させる手順を、ステップごとに解説するものです。順番通りに進めてください。各ステップは前のステップの結果を前提としています。
>
> **質問する前に必ずこのドキュメントを読んでください。** これまで受けた「モデルがロードされない」系の質問は、全てこのガイドのいずれかのステップを飛ばしていました。

---

## 前提条件

- [ ] 動作するポリシーモデルがある (PyTorch, JAX, いずれでも可)
- [ ] GPU ≥ 16 GB VRAM、Docker、NVIDIA Container Toolkit が入ったマシンがある
- [ ] [README.md](../README.md) §5 (WebSocket I/O Contract) と §6 (環境変数) を読んだ
- [ ] **手を加えていない** ベースリポジトリに対して、デフォルト OpenPI チェックポイントで `./RUN-DOCKER-CONTAINER.sh up` が動作し、ログに `Action executed.` が出ることを確認した。(自分のコードを追加する前にホスト環境が正しいことを確認します)

---

## Step 0. Fork してブランチを切る

**提出は `airoa-org/airoa-evaluation-ICRA` のフォークから行ってください。** 別リポジトリからではなく、フォークから。ハーネス (`RUN-DOCKER-CONTAINER.sh`, `docker-compose.yml`, `runtime_core/`, `packages/`, `deploy/`) はここにしか無いので、別リポジトリでは評価できません。

**スタート地点となるブランチを正しく選んでください:**

| ブランチ | こういう時に使う |
|---|---|
| **`base`** *(推奨デフォルト)* | PyTorch / JAX / LeRobot / 独自フレームワークを使う場合。最小ハーネス + `ZeroPolicy` プレースホルダーで、smoke test が即通る状態。 |
| **`sample-openpi`** | モデルが `PI0Pytorch` / OpenPI 互換で、OpenPI ローダの動作例を参考にしたい場合。`src/openpi/` 一式が同梱されている。 |

```bash
# GitHub上で airoa-org/airoa-evaluation-ICRA を fork → <あなたのorg>/airoa-evaluation-ICRA
git clone https://github.com/<あなたのorg>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # OpenPI 例を見たい場合のみ `sample-openpi`
git checkout -b feat/my-policy     # 自分の提出用ブランチ
```

---

## Step 1. `src/` 以下にモデルコードを配置

```
src/
└── my_policy/
    ├── __init__.py
    ├── model.py              # ネットワーク
    ├── adapter.py            # WebSocket契約とモデルを繋ぐ層
    └── (関連ファイル)
```

独自のローカルパッケージ (例: カスタムデータローダ) が必要なら `packages/` 以下に追加し、`pyproject.toml` のworkspaceに登録してください。具体的な構造は §9 のレイアウト例を参照してください。

---

## Step 2. アダプタを書く

パイプラインは皆さんのポリシーオブジェクトに対して、以下のメソッドのみを呼び出します:

```python
policy.infer(obs: dict) -> dict
```

これだけです。`reset()`, `predict()`, `__call__` は呼ばれません。ダックタイピングで十分で、基底クラス継承は不要です (`policy_client.base_policy.BasePolicy` を使ってもよい)。

最小アダプタテンプレート:

```python
# src/my_policy/adapter.py
import numpy as np
import torch
import os
from .model import MyModel

class MyPolicyAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = MyModel(...).to(self.device).eval()

        # checkpoint_path は「ディレクトリ」。この中から重みを読み込む
        weights = os.path.join(checkpoint_path, "model.pt")
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state.get("model_state_dict", state), strict=False)

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        # 契約 (README §5):
        head_rgb = obs["head_rgb"]                 # (480, 640, 3) uint8
        hand_rgb = obs["hand_rgb"]                 # (480, 640, 3) uint8
        state    = obs["state"].astype(np.float32) # (8,) float32
        prompt   = obs.get("prompt", "")           # str

        actions = self._run(head_rgb, hand_rgb, state, prompt)   # forward
        actions = np.asarray(actions, dtype=np.float32)          # (T, 11) float32
        assert actions.ndim == 2 and actions.shape[1] == 11 and actions.shape[0] >= 1
        return {"actions": actions}
```

**よくある間違い**:
- `{"actions": (11,)}` (1次元)を返す。必ず `(T, 11)` の2次元 (たとえ `T=1` でも)
- 32次元のアクションをゼロパディングして返す。厳密に11次元、README §5 の順で
- `obs["task_index"]` を読む。そんなキーはありません。タスク記述は `obs["prompt"]` (文字列)。モデルがタスクIDを必要なら `infer` の中で `prompt → task_id` の対応を作る
- `state` を32次元として受け取る。**常に8次元**

---

## Step 3. `serve_hsr_policy_ws.py` にアダプタを組み込む

ベースのファイルはデフォルトの OpenPI ローダを使っています。`PI0Pytorch` 以外のモデルではこのブロックを自分のアダプタに置き換えます。

**元のコード (抜粋)** — `server/serve_hsr_policy_ws.py`:
```python
from openpi.policies.policy_config import create_trained_policy
...
policy = create_trained_policy(...)
server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata={})
server.serve_forever()
```

**差し替え (例)**:
```python
from my_policy.adapter import MyPolicyAdapter
...
policy = MyPolicyAdapter(
    checkpoint_path=args.checkpoint_dir,
    device=args.pytorch_device or "cuda",
)
server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata={})
server.serve_forever()
```

ファイルの残り (引数パース、ロギング、`WebsocketPolicyServer` のインスタンス化) はそのまま残してください。サーバ側が要求するのは `policy` に `.infer(obs)` があることだけです。

> 🔍 環境変数 `POLICY_BACKEND` でバックエンド (OpenPI / LeRobot / 独自) を切り替える設計も有効です。`serve_hsr_policy_ws.py` の冒頭で env var を読み、対応するローダを選択する形にできます。

---

## Step 4. `server/Dockerfile` で依存をインストール

Python 依存は `server/Dockerfile` に追加してコンテナに焼き込んでください。

diff 例:

```dockerfile
# server/Dockerfile
...
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        torch==2.5.1 \
        transformers==4.46.0 \
        your-custom-dep==1.2.3
```

バージョンはピン留めしてください。評価マシンは評価中ネットワークが制限されることがあり、`pip install` が途中失敗するとそのまま評価失敗になります。

> ⚠️ **Blackwell GPU (RTX 5070 Ti)** は CUDA 12.8+ とそれに対応した torch ビルドが必要です。[§8](#8-ハードウェア固有-blackwellhuggingface-等) 参照。

`uv` で依存を追加したら lock を更新:
```bash
uv lock
git add uv.lock pyproject.toml
```

---

## Step 5. チェックポイントを「ディレクトリ」としてパッケージ化

ハーネスは `POLICY_CHECKPOINT_PATH` を**ディレクトリ**として検証します:

```bash
# RUN-DOCKER-CONTAINER.sh:
if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
fi
```

`.pt` ファイル単体を指定すると即エラーです。以下のようなディレクトリ構造にしてください:

```
my_checkpoint/
├── model.pt                   # or model.safetensors
├── config.json
├── tokenizer/                 # 必要なら
└── normalization_stats.json   # 必要なら
```

アダプタの中で `os.path.join(checkpoint_path, "model.pt")` のように読み出します。

---

## Step 6. ローカル smoke test (必須)

```bash
cd airoa-evaluation-ICRA   # 自分のfork、提出ブランチ上
export POLICY_CHECKPOINT_PATH=/abs/path/to/my_checkpoint
export POLICY_PYTORCH_DEVICE=cuda

# 1. ビルド & 起動 (TEST_MODE=true デフォルト、実機不要)
./RUN-DOCKER-CONTAINER.sh up

# 2. サーバがモデルをロードできたか確認
./RUN-DOCKER-CONTAINER.sh logs policy_server
# 期待: "server listening on 0.0.0.0:8000"

# 3. 合成観測を流す
./RUN-DOCKER-CONTAINER.sh shell
# コンテナ内:
roslaunch hsr_policy_client hsr_policy_client.launch
# policy_server のログに "Action executed." が (繰り返し) 出ることを期待

# 4. 停止
./RUN-DOCKER-CONTAINER.sh down
```

**確認ポイント**:
- チェックポイントがロードできる (Dockerfileの依存が正しい、`adapter.__init__` が動く)
- 合成観測に対して `infer()` が正しい shape を返す

ここで失敗したら本番評価も同じように失敗します。**smoke test を通さずに提出しないでください**。

---

## Step 7. 提出用 note を書く

参加者の提出形式は **fork URL + ブランチ + note** (実行方法の短い説明) です。ビルドの記憶が新しいうちに note を書いてしまいましょう。

推奨する書き方:

1. [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md) を repo root に `REPRODUCTION_STEPS.md` としてコピー
2. 全セクション (概要、前提、実行コマンド、env var、編集したファイル、特記事項、期待出力) を埋める
3. 提出ブランチにコミット
4. 提出時にそのファイルへのリンクを note として貼る (または本文を転記)

形式は必須ではありませんが、これが初回再現成功の一番の近道です。自由記述の note にする場合でも、テンプレートの各項目を全てカバーするようにしてください。

---

## Step 8. チェックポイントを S3/Wasabi にアップロード

```bash
aws --profile <プロファイル名> \
    --endpoint-url <運営から指定されたURL> \
    s3 sync /abs/path/to/my_checkpoint/ \
    s3://<バケット名>/<パス>/

# 確認
aws --profile <プロファイル名> \
    --endpoint-url <URL> \
    s3 ls s3://<バケット名>/<パス>/
```

必要ファイルが全部入っていること、サイズが一致することを目視確認してください。よくあるミス: 大きな `.safetensors` がリトライの末に欠けたままアップロード完了扱いになる。

---

## Step 9. 提出

運営に以下の3つを共有してください:

1. **fork URL** — 例: `https://github.com/<あなた>/airoa-evaluation-ICRA`
2. **ブランチ名** — 例: `feat/my-policy`
3. **note** — 実行方法の説明: S3チェックポイントパス、必要な env var、特記事項 (同梱tokenizer、非標準のGPUメモリ要件など)。Step 7 でコミットした `REPRODUCTION_STEPS.md` へのリンクを貼るのが最もきれい。

任意 (精度のため推奨): **コミットハッシュ** (`git rev-parse HEAD`) を note に含めておくと、評価者がテスト済みの状態を正確にチェックアウトできます (後から追加コミットしても影響を受けません)。

---

## 8. ハードウェア固有 (Blackwell 等)

評価マシンの GPU は **RTX 5070 Ti (Blackwell, sm_120)** です。Ada/Ampere/Hopper 専用ビルドの wheel をそのまま使うと、実行時に `no kernel image is available for execution on the device` でエラーとなります。

### Blackwell 対応の PyTorch/CUDA

CUDA 12.8+ と対応 Torch が必要。実用的な方法:

**(a)** `server/Dockerfile` のベースを CUDA 12.8+ に:
```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```
対応する Torch wheel をインストール:
```dockerfile
RUN uv pip install --system \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128
```

**(b)** クライアント側でも GPU が必要な構成 (モデルが client 側でも走るケース) では、`client/Dockerfile` も同様に変更。**これは `client/Dockerfile` を編集してよい数少ないケース**。提出 note に理由を明記してください。

### モデルコンパイル (torch.compile / JAX JIT)

初回推論が重い (>30秒) 実装では、コンパイルキャッシュを永続パスに固定するか、コンパイル自体を無効化してください。評価者は初回推論に 300秒のタイムアウトを設けています。超えるとその run は中止扱い。

---

## 9. レイアウト例

以下は、きれいに適応された fork の匿名の一例です。構造の参考にしてください (実際のファイル名・ディレクトリ名は皆さんの実装次第です)。

```
airoa-evaluation-ICRA/
├── server/
│   ├── Dockerfile                      # 改: CUDA ベース、独自の依存追加
│   ├── entrypoint.sh                   # 改: バックエンド切替 (任意)
│   ├── serve_hsr_policy_ws.py          # 改: 自分のポリシーをロード
│   ├── my_policy_loader.py             # モデルをインスタンス化する薄いラッパー
│   └── my_controller.py                # 任意: 高レベル制御ロジック
├── src/
│   ├── openpi/…                        # ベースそのまま
│   └── my_policy/                      # 皆さんの新規コードはここ
├── packages/policy-client/              # 変更なし (protocol)
├── docker-compose.yml                   # + サーバが必要とする env var
├── client/Dockerfile                    # ハードウェア互換のためだけに編集
├── .env                                 # 各種 env var のローカルデフォルト
├── task_config.json                     # 任意: タスク固有の設定
├── controller_config.yaml               # 任意: コントローラ設定
├── tokenizer/<your_tokenizer>/          # 同梱、実行時に外部認証不要
└── REPRODUCTION_STEPS.md                # 評価者がそのまま実行する手順書
```

ポイント:

- **新規コードは `src/<your_policy>/` に集約** — `server/` 内に散らさない
- **`server/` は薄いラッパー** で `src/…` を呼ぶだけ。レビューしやすい
- **設定は YAML/JSON + `.env` で外出し**、評価者がコードを読まずに再現できる
- **依存 (tokenizer, アダプタ等) を同梱**、外部認証不要
- **`REPRODUCTION_STEPS.md` が最小限でコピペ可能** — これが初回再現成功の最大の要因

---

## 10. よく見る落とし穴

1. **別リポジトリを提出する。** 本リポジトリのforkから提出してください。スタンドアロン repo にはハーネスがありません
2. **メソッド名違い。** `infer` です。`predict`, `__call__` ではありません
3. **state の次元違い。** 8次元、32次元ではありません
4. **シングルステップ action。** `(T, 11)` を返す、`(11,)` ではない
5. **`.pt` ファイル単体を `POLICY_CHECKPOINT_PATH` に指定。** ディレクトリにしてください
6. **独自の env var 名** (`MODEL_CHECKPOINT`, `DEVICE`, `STATE_DIM`)。README §6 の3つしか読まれません
7. **理由を明記せずに `client/Dockerfile` を編集。** Blackwell CUDA 対応ならOKですが提出 note に理由を書いてください
8. **smoke test をしない。** 手元のPCで `./RUN-DOCKER-CONTAINER.sh up` + `roslaunch` が `Action executed.` を出さなければ、本番も同じ失敗をします
9. **チェックポイントのアップロード漏れ。** `sync` 後に `aws s3 ls` で全ファイル・サイズを確認
10. **ホスト Python への依存。** サーバコンテナ内で完結させてください。評価者はホストに何もインストールしません

---

## 11. 次に読むもの

- ❓ 具体的なエラー? → [FAQ](FAQ.md) ([English](FAQ.md))
- 📝 再現手順書を書く? → [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md)
- 🏷 ベース repo README? → [../README.md](../README.md)
