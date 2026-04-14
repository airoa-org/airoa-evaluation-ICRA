# FAQ / Errata

> 公開リポジトリ `airoa-org/airoa-evaluation-ICRA` の README に対する補足を1箇所にまとめ、全参加チームに配布するドキュメントです。評価プロトコルを変更するものではなく、複数チームから質問が寄せられた点の再記述と、提出時によくあるミスへの対処をまとめています。

リポジトリ内の根拠:
- [README.md](../README.md) §3 (編集可能スコープ), §5 (WebSocket I/O Contract), §6 (環境変数)
- [INTEGRATION_GUIDE_ja.md](INTEGRATION_GUIDE_ja.md) — ステップバイステップ
- `server/serve_hsr_policy_ws.py`, `server/entrypoint.sh`, `server/Dockerfile`
- `RUN-DOCKER-CONTAINER.sh`
- `deploy/hsr_policy_client/launch/hsr_policy_client.launch`

---

## 0. どのブランチから fork するか

参加者向けに2つのブランチがあります。スタック (使う技術) に合うものを選んでください — 違うブランチを選ぶと「ハーネスの剥がし作業」に数日浪費する原因になります。

| ブランチ | 同梱されるもの | こういう場合に選ぶ |
|---|---|---|
| **`base`** *(デフォルトのスタート地点)* | 最小ハーネス + `ZeroPolicy` プレースホルダ。`src/` は空 | PyTorch / JAX / LeRobot / 独自フレームワークを使う場合 — つまり **大多数の参加者** |
| **`sample-openpi`** | 上記 + `serve_hsr_policy_ws.py` に OpenPI ローダ配線済 + `src/openpi/` 一式。`POLICY_CONFIG_NAME` が必要 | `PI0Pytorch` 互換のモデルを統合する場合で、OpenPI ローダを動作例として参考にしたい |

OpenPI を能動的に使わないなら **`base` から fork** してください。`sample-openpi` には ~14k 行の OpenPI コードが入っており、そうでないチームはこれを削除する作業から始めることになります。

```bash
git clone https://github.com/<あなた>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout base                  # OpenPI 例を見たい場合のみ `sample-openpi`
git checkout -b feat/my-policy
```

---

## 1. 編集可能スコープ

提出は `airoa-org/airoa-evaluation-ICRA` のフォークから行ってください (どのブランチからかは §0 参照)。**コア** の編集可能スコープ:

- `server/` — `serve_hsr_policy_ws.py`, `Dockerfile`, `entrypoint.sh` を含む
- `src/` — 皆さんのモデル/ポリシーコード

**真に必要な場合のみ** 編集してよいパス (利便性だけでの編集は不可):

- `docker-compose.yml` — サーバが必要とする env var やボリュームの追加
- `client/Dockerfile` — **ハードウェア互換性のためのみ** (例: Blackwell 用 CUDA ベースイメージ)
- `packages/<独自パッケージ>/` — 追加のローカルパッケージ
- `pyproject.toml`, `uv.lock` — 依存追加

**変更してはいけない** パス:

- `runtime_core/` — WebSocket サーバプロトコル
- `packages/policy-client/` — WebSocket クライアントプロトコル
- `deploy/hsr_policy_client/` — ROS クライアント実装
- `RUN-DOCKER-CONTAINER.sh` — ハーネスのエントリポイント

**モデルだけを含むスタンドアロンリポジトリは単独では評価できません** — ハーネスはベースリポジトリにしか存在しないためです。§6 参照。

「条件付きで許可」のパスを編集したら、提出 note (→ [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md)) に理由を明記してください。

---

## 2. PyTorch はサポートされています

パイプラインは JAX 専用ではありません。PyTorch モデル向けの2経路:

**(a) デフォルトの `openpi` ローダ。** `serve_hsr_policy_ws.py` が `openpi.policies.policy_config.create_trained_policy(...)` を呼び、チェックポイント内の `model.safetensors` で PyTorch 自動判定。`src/openpi/models_pytorch/` の `PI0Pytorch` アーキテクチャのみ対応。

**(b) カスタムアダプタ (`PI0Pytorch` 以外で推奨)。** `server/serve_hsr_policy_ws.py` を編集して、自分のポリシークラスをインスタンス化し `WebsocketPolicyServer` に渡します。サーバ要件は以下のみ:

```python
policy.infer(obs: dict) -> dict
```

ダックタイピングで十分、継承は任意 (`policy_client.base_policy.BasePolicy`)。詳細は [INTEGRATION_GUIDE_ja §2-3](INTEGRATION_GUIDE_ja.md#step-2-アダプタを書く)。

---

## 3. WebSocket I/O Contract (README §5 の再記述)

### 3.1 `policy.infer(obs)` に渡される観測 dict

```python
{
  "head_rgb": np.ndarray,  # shape (480, 640, 3),  dtype uint8
  "hand_rgb": np.ndarray,  # shape (480, 640, 3),  dtype uint8
  "state":    np.ndarray,  # shape (8,),           dtype float32
  "prompt":   str,
}
```

`state` は **8次元**、順序:

```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint]
```

通信は WebSocket 上の msgpack-numpy。配列は `np.ndarray` として既にデシリアライズ済み。

### 3.2 `policy.infer(obs)` から返すアクション dict

```python
{"actions": np.ndarray}   # shape (T, 11), dtype float32, T >= 1, 全て有限値
```

アクションの順序 (11次元):

```
[arm_lift_joint, arm_flex_joint, arm_roll_joint,
 wrist_flex_joint, wrist_roll_joint, gripper,
 head_pan_joint, head_tilt_joint,
 base_x, base_y, base_t]
```

`T >= 1` なら何でもOK。シングルステップの場合は **`(1, 11)` の2次元配列**にしてください、`(11,)` は不可。

### 3.3 エピソード境界

サーバはポリシーの `reset()` を呼びません。エピソードごとの状態が必要なら、`infer()` 内で新しい prompt (または無通信ギャップ) を検知して内部でリセットしてください。

### 3.4 `obs` に **無い** もの

- `task_index` / `task_id` はありません。必要なら `prompt` から推論してください
- `episode_step` / `timestep` もありません。必要なら内部でトラッキング
- 深度画像、点群はありません (RGBカメラのみ)

---

## 4. パイプラインの動かし方

### 4.1 `POLICY_CHECKPOINT_PATH` は**ディレクトリ**でなければならない

`RUN-DOCKER-CONTAINER.sh` の検証:

```bash
if [[ ! -d "${POLICY_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] POLICY_CHECKPOINT_PATH does not exist: ${POLICY_CHECKPOINT_PATH}"
    exit 1
fi
```

`.pt` / `.safetensors` ファイル単体では即エラー。ディレクトリにパッケージングし、アダプタの中で重みファイルを読み出してください。

### 4.2 `server/entrypoint.sh` が実際に読む環境変数

| 変数 | 目的 |
|---|---|
| `POLICY_CHECKPOINT_PATH` | サーバに `--checkpoint-dir` として渡されるディレクトリ |
| `POLICY_PYTORCH_DEVICE` | 任意; `--pytorch-device` (例: `cuda`) |
| `POLICY_CONFIG_NAME` | 任意; デフォルト OpenPI ローダ用の config 名 |

これ以外の名前 (`MODEL_CHECKPOINT`, `DEVICE`, `STATE_DIM`, `ACTION_DIM` 等) は**読まれません**、効果ゼロ。

サーバ実装が追加の env var (例: `POLICY_BACKEND=lerobot`) を必要とするなら、**自分の** `docker-compose.yml` に追記し (ローカル smoke test 用に `.env` も)、提出 note に列挙してください。

### 4.3 どのコンテナがチェックポイントをロードするか

`docker compose` は2つのコンテナを起動:

- `airoa_policy_server` — `POLICY_CHECKPOINT_PATH` からモデルをロードし WebSocket サーバを走らせる
- `airoa_hsr_client` — 観測を集めアクションを実行する ROS ノード

チェックポイントをロードするのは **サーバ**、クライアントではありません。`roslaunch hsr_policy_client hsr_policy_client.launch` は `deploy/hsr_policy_client/launch/hsr_policy_client.launch` で宣言された引数 (`policy_server_host`, `policy_server_port`, `test_mode` 等) しか受け付けません。`checkpoint:=…` や `device:=…` のような未知の引数は ROS が静かに捨てます。

### 4.4 ビルドされる Dockerfile はどれか

`./RUN-DOCKER-CONTAINER.sh up` は `docker compose up --build` を走らせ、**皆さんの fork 内の** `server/Dockerfile` (と `client/Dockerfile`) をビルドします。別リポジトリに置いた Dockerfile はビルドされません。Python 依存は fork 内の `server/Dockerfile` を編集して入れてください。

---

## 5. S3 アップロード前のローカル smoke test

```bash
cd airoa-evaluation-ICRA   # fork、提出ブランチ
export POLICY_CHECKPOINT_PATH=/abs/path/to/your_checkpoint_dir
export POLICY_PYTORCH_DEVICE=cuda
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch    # test_mode=true がデフォルト
./RUN-DOCKER-CONTAINER.sh logs policy_server            # "Action executed." を期待
```

`test_mode=true` ではクライアントが §3.1 の形式の合成観測を送ります。ここで shape/キー/dtype エラーが出るなら、本番評価も同じように失敗します。

---

## 6. よくあるミス

### 6.1 スタンドアロンリポジトリから提出

**症状.** 評価者から `./RUN-DOCKER-CONTAINER.sh: No such file or directory` や、ビルド不能の報告。

**原因.** `<your-org>/MyPolicy` のようなモデルコードだけの repo から提出している。ハーネス (`RUN-DOCKER-CONTAINER.sh`, `docker-compose.yml`, `runtime_core/`, `packages/`) は `airoa-org/airoa-evaluation-ICRA` のforkにしかありません。

**対処.** このリポジトリをfork、`src/` にモデルコード配置、`server/` 編集、それを提出。

### 6.2 メソッド名違い (`predict`, `__call__`)

**症状.** チェックポイントはロードされるが初回推論で `AttributeError: 'MyAdapter' object has no attribute 'infer'`。

**対処.** メソッド名を `infer` に。他は呼ばれません。

### 6.3 state の次元違い (8 ではなく 32)

**症状.** アダプタ内の `ValueError: could not broadcast` や shape 不一致。

**対処.** `obs["state"]` は `(8,)` float32 で来ます。訓練が32次元想定なら `infer()` 内で射影/パディングしてください。

### 6.4 シングルステップ `(11,)` を返してしまう

**症状.** infer後にサーバエラー、クライアントが実行拒否。

**対処.** 2次元配列を返す。最小有効形は `(1, 11)` で `(11,)` は不可。

### 6.5 独自 env var 名は効かない

**症状.** `MODEL_CHECKPOINT=...`, `DEVICE=cuda` を設定したのにサーバがチェックポイントを見つけられない。

**対処.** `POLICY_CHECKPOINT_PATH`, `POLICY_PYTORCH_DEVICE`, `POLICY_CONFIG_NAME` のみ読まれます。リネームしてください。

### 6.6 `.pt` ファイル単体を `POLICY_CHECKPOINT_PATH` に

**症状.** `[ERROR] POLICY_CHECKPOINT_PATH does not exist: /path/to/model.pt`。

**対処.** ディレクトリ化してアダプタの中からファイルを読む。

### 6.7 `server/Dockerfile` を編集し忘れ

**症状.** サーバ起動時に自分の依存で `ImportError` / `ModuleNotFoundError`。

**対処.** 全部 `server/Dockerfile` でインストール。評価マシンはイミュータブル扱い、Docker イメージ外のものは使えません。

### 6.8 提出前に smoke test していない

**症状.** 評価側の実行で、smoke test してれば気づいたはずのエラー。

**対処.** 先に smoke test を必ず通す。通らないコードを提出しても本番で通ることはありません。

### 6.9 Gated な HuggingFace モデルが `HF_TOKEN` を要求

**症状.** `OSError: You are trying to access a gated repo`。

**対処.** `HF_TOKEN` が設定されている前提にしないでください。イメージビルド時に `COPY` で焼き込むか、チェックポイントディレクトリに同梱。[INTEGRATION_GUIDE_ja §8](INTEGRATION_GUIDE_ja.md#huggingface-トークン) 参照。

---

## 7. ハードウェア固有: Blackwell (RTX 5070 Ti)

評価用 GPU は Blackwell (compute capability 12.0, sm_120)。CUDA 12.8+ と対応 Torch wheel が必要で、古い wheel では `no kernel image is available for execution on the device` で失敗します。

- `server/Dockerfile` (モデルがクライアントでも走るなら `client/Dockerfile` も) のベースを `nvidia/cuda:12.8.1-*` に
- `torch` は `--index-url https://download.pytorch.org/whl/cu128` でインストール
- 動作例は [INTEGRATION_GUIDE_ja §8](INTEGRATION_GUIDE_ja.md#blackwell-対応の-pytorchcuda)

---

## 8. 提出前チェックリスト

提出前に必ず確認:

- [ ] `airoa-org/airoa-evaluation-ICRA` の fork のブランチから提出している (別リポジトリではない)
- [ ] `server/serve_hsr_policy_ws.py` が自分のポリシーをインスタンス化し `WebsocketPolicyServer` に渡している
- [ ] ポリシーオブジェクトが `infer(obs: dict) -> dict` を公開している
- [ ] `infer` が `head_rgb`, `hand_rgb`, `state` (shape `(8,)`), `prompt` (str) を受け取る
- [ ] `infer` が `{"actions": np.ndarray}` を返し、`actions.shape == (T, 11)`, `dtype=float32`, `T >= 1`, 全て有限
- [ ] チェックポイントが**ディレクトリ**。アダプタがその中からファイルを読む
- [ ] 依存は `server/Dockerfile` に pin 留め済み。ホスト Python に依存しない
- [ ] GPU を使うなら Blackwell 対応の CUDA + Torch wheel
- [ ] ローカル smoke test が通る: `./RUN-DOCKER-CONTAINER.sh up` → `roslaunch hsr_policy_client hsr_policy_client.launch` → `Action executed.`
- [ ] チェックポイントが S3 に全部アップロード済み (`aws s3 ls` と合計サイズで検証)
- [ ] 提出用 note を準備 — 実行方法 (S3 チェックポイントパス、env var、特記事項) をカバー。[REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md) の構造で書き、`REPRODUCTION_STEPS.md` としてコミットしそのリンクを note とするのが最もきれい
- [ ] 提出連絡に: **fork URL**、**ブランチ名**、**note** (コミットハッシュは任意、note に含めると精度が上がる)

---

## 9. 提出フロー (要約)

1. `airoa-org/airoa-evaluation-ICRA` を fork
2. `src/…` にモデルコード配置
3. `server/serve_hsr_policy_ws.py` を編集 (非 `PI0Pytorch` の場合)
4. `server/Dockerfile` / `server/entrypoint.sh` に依存追加
5. チェックポイントをディレクトリ化
6. §5 の smoke test を実行
7. 提出用 note を書く (推奨: [REPRODUCTION_STEPS.template_ja.md](REPRODUCTION_STEPS.template_ja.md) を埋めて `REPRODUCTION_STEPS.md` としてコミット)
8. チェックポイントを S3 にアップロード
9. **fork URL + ブランチ + note** を運営に提出

---

追加の質問は運営の指定チャンネルで共有してください。FAQ に追記して全員に反映します。
