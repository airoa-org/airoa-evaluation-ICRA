# REPRODUCTION_STEPS — Team XX (提出用 note テンプレート)

> 提出は **fork URL + ブランチ + note** (実行方法の説明) の形で運営に送ります。このファイルは note の推奨構造です。リポジトリ直下に `REPRODUCTION_STEPS.md` としてコピーし、全セクションを埋めてコミットし、そのリンク (または本文) を note として送ってください。評価者はこの通りに実行するので、コマンドとパスは正確に。括弧内の文言は記入ガイドなので実際の値に書き換えてください。
>
> English version: [REPRODUCTION_STEPS.template.md](REPRODUCTION_STEPS.template.md)

---

## 1. 概要

| 項目 | 値 |
|---|---|
| モデル概要 | (一行で — 何のモデルか。例: 「π0.5 SFT (public_tasks/task6911 で fine-tune)」) |
| フレームワーク | (例: OpenPI / LeRobot / 独自 PyTorch / 独自 JAX) |
| リポジトリ | `https://github.com/<あなた>/airoa-evaluation-ICRA` |
| ブランチ | `feat/my-policy` |
| コミットハッシュ | `abcd1234…` (任意、含めると精度が上がる) |
| チェックポイント S3 パス | `s3://<bucket>/<path>/` |
| 想定 VRAM | (例: ~9 GB) |

---

## 2. 前提条件

- NVIDIA GPU ≥ 16 GB VRAM (**Blackwell 対応必須**: RTX 5070 Ti / compute 12.0)
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- 外部認証情報 (例: `HF_TOKEN`, S3 credentials) が必要か? **不要なら「不要」と明記してください**

> repo/image のビルド時に同梱されていないリソース (gated HF モデル、外部サービス等) に依存する場合、評価者がアクセスする必要のあるもの全てを列挙してください。評価環境はオフライン/制限されている場合があります。

---

## 3. 再現手順

### 3.1 Clone とチェックアウト

```bash
git clone https://github.com/<あなた>/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout <ブランチ>
git rev-parse HEAD  # 任意: §1 のハッシュと一致すること
```

### 3.2 チェックポイントのダウンロード

```bash
mkdir -p checkpoints/<名前>
aws --profile <プロファイル名> --endpoint-url <URL> \
    s3 sync s3://<バケット名>/<パス>/ checkpoints/<名前>/
```

(実際に使うダウンロード方法に合わせて置き換えてください — HF CLI, `curl` 等)

### 3.3 環境変数

ハーネスが読むのは以下の3つだけです。使うものを列挙してください:

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoints/<名前>       # ディレクトリ必須
export POLICY_PYTORCH_DEVICE=cuda                             # 任意
export POLICY_CONFIG_NAME=<openpi_config_name>                # デフォルト OpenPI ローダを使う場合のみ
```

サーバ実装が他の env var を読むなら、**全て**ここに列挙し、自分の `docker-compose.yml` / `.env` で設定してください。

### 3.4 コンテナ起動

```bash
./RUN-DOCKER-CONTAINER.sh up
```

### 3.5 動作確認

```bash
# ポリシーサーバが起動するまで待つ:
until curl -s http://localhost:8000/healthz 2>/dev/null | grep -q OK; do sleep 5; done
echo "READY"

# GPU 使用量の確認:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### 3.6 停止

```bash
./RUN-DOCKER-CONTAINER.sh down
```

---

## 4. ベースリポジトリからの変更点

コアのスコープ (`server/`, `src/`) を超えて編集したパスを列挙し、理由を説明してください。

| パス | 理由 |
|---|---|
| `server/serve_hsr_policy_ws.py` | (例: デフォルト OpenPI ローダではなく MyPolicyAdapter をロード) |
| `server/Dockerfile` | (例: torch 2.5.1 + transformers 4.46.0 + 独自依存を追加) |
| `src/<your_policy>/` | (例: 新規モデル実装) |
| `docker-compose.yml` | (例: サーバが読む env var を1つ追加) |
| `client/Dockerfile` | (例: Blackwell 対応のため CUDA 12.8.1 ベースに変更) |

---

## 5. 重要な注意事項

上記コマンドから自明ではないが評価者が知っておくべき事項を列挙してください。例:

- (例: 「初回起動は重みロードに ~2 分かかる」)
- (例: 「tokenizer は `<パス>` に同梱済み、`HF_TOKEN` は不要」)
- (例: 「モデルのコンパイルを無効化、300秒の初回推論タイムアウトを避けるため」)

---

## 6. チェックポイントのファイル構成

チェックポイントディレクトリの中身を記載してください。構成はフレームワーク次第ですが、代表的な2パターン:

**OpenPI スタイル:**

```
<名前>/
├── params/                            # または model.safetensors
├── assets/<asset_id>/
│   └── norm_stats.json
└── config.yaml                        # または config.json
```

**独自 PyTorch スタイル:**

```
<名前>/
├── model.pt                           # または model.safetensors
├── config.json
└── (必要なら tokenizer / preprocessor ファイル)
```

実際の構成と合計サイズを貼り付けてください:

```
(`tree` または `ls` 出力を貼る)
```

合計サイズ: ~X GB

---

## 7. smoke test の期待出力

自分の成功した run から1-3行コピーして、「動いている」とはどういう状態かを評価者に示してください:

```
(実際のログを貼る、例:)
[INFO] server listening on 0.0.0.0:8000
[INFO] Action executed.
[INFO] Action executed.
```

---

## 8. 連絡先

- チーム: (team name)
- 代表者: (name) <email>
- 提出日: YYYY-MM-DD
