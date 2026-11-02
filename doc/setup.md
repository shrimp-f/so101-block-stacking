# Docker 環境構築手順（Windows 10 + Docker Desktop）

対象ホスト: Windows 10 Home 22H2 (19045) / GeForce GTX 1070 / RAM 16GB

## 1. 事前確認

### 1.1 仮想化が有効か
タスクマネージャー → 「パフォーマンス」→「CPU」の右下にある **「仮想化: 有効」** を確認する。
「無効」の場合は BIOS/UEFI の設定で Intel VT-x（Intel Virtualization Technology）を有効にする。

### 1.2 NVIDIA ドライバ
現在のドライバ 560.94 は WSL2 の CUDA に対応済み。問題が出たら GeForce Experience か
NVIDIA 公式サイトから最新の Game Ready / Studio ドライバに更新する。
**WSL 内に Linux 用の NVIDIA ドライバを入れる必要はない（入れてはいけない）。**

## 2. WSL2 の有効化

**管理者として** PowerShell を開き、次を実行する。

```powershell
wsl --install --no-distribution   # WSL と仮想マシン基盤を有効化（ディストリビューションは入れない）
wsl --update                      # WSL カーネルを最新化
wsl --set-default-version 2       # 既定を WSL2 にする
```

実行後、**Windows を再起動**する。

> 既存の Ubuntu-18.04（WSL1）はそのまま残る。Docker Desktop は専用の WSL2 環境を
> 自動で作るので、Ubuntu 側の作業は不要。

## 3. Docker Desktop のインストール

1. https://www.docker.com/products/docker-desktop/ から **Download for Windows – AMD64** をダウンロード
2. `Docker Desktop Installer.exe` を実行
3. 設定画面で **「Use WSL 2 instead of Hyper-V」にチェック**を入れたまま進める
4. インストール完了後、サインアウトまたは再起動を求められたら従う
5. Docker Desktop を起動し、利用規約に同意する（個人利用なので Personal で可。サインインは省略してよい）

## 4. Docker Desktop の設定

- **Settings → General**: 「Use the WSL 2 based engine」にチェックが入っていることを確認
- **Settings → Resources → Advanced → Disk image location**（任意）:
  C ドライブの空きが少ない場合、イメージの保存先を別ドライブに変更する
  （LeRobot + CUDA のイメージは 10〜15GB 程度になる見込み）

### 4.1 WSL2 のメモリ上限（推奨）
WSL2 モードでは CPU とメモリの上限を `.wslconfig` で指定する。
`%USERPROFILE%\.wslconfig` を作成し、次の内容を書く。

```ini
[wsl2]
memory=12GB
processors=6
swap=8GB
```

保存後、PowerShell で `wsl --shutdown` を実行し、Docker Desktop を再起動する。

## 5. 動作確認

PowerShell（管理者でなくてよい）で順に実行する。

```powershell
docker version                     # Client と Server の両方が表示されれば OK
docker run --rm hello-world        # "Hello from Docker!" が表示されれば OK
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

最後のコマンドで **GTX 1070 の情報が表示されれば、コンテナから GPU を使える**。

## 6. シミュレータ用コンテナ

プロジェクトのフォルダで実行する。

```powershell
docker compose build                                 # 初回は 15〜20 分ほどかかる
docker compose run --rm sim python scripts/check_env.py   # 環境チェック
docker compose run --rm sim                          # コンテナ内の bash に入る
```

`check_env.py` で次が表示され、`outputs/check_render.png` に 2 個のブロックの画像が出れば OK。

```
lerobot 0.6.1
torch 2.11.0+cu126 / CUDA available: True
  device: NVIDIA GeForce GTX 1070, capability: (6, 1)
mujoco 3.17.0 / MUJOCO_GL=egl
```

ブラウザで MuJoCo viewer を操作したいときは `START_VNC=1` を付けて起動し、
http://localhost:6080/vnc.html を開く。

```powershell
$env:START_VNC=1; docker compose run --rm --service-ports sim
```

### 構成メモ
- ベースイメージは `nvidia/cuda:12.6.3-base-ubuntu24.04`。lerobot 0.6 系は Python 3.12 以上が必要なため Ubuntu 24.04 を使う
- PyTorch は cu126 ビルドを使う。GTX 1070（sm_61）は sm_60 のバイナリで動作する
- 学習（PyTorch）は GPU で動く。一方、MuJoCo の描画は WSL2 上の Docker では NVIDIA の EGL が使えないため、
  **CPU のソフトウェア描画（Mesa llvmpipe）** になる。物理計算はもともと CPU なので、影響するのは描画速度だけ
- データセットは Docker ボリューム `so101-data`（コンテナ内 `/workspace/data`）に、
  HuggingFace のキャッシュは `hf-cache` に保存する

## 7. トラブルシューティング

| 症状 | 対処 |
|---|---|
| Docker Desktop 起動時に "WSL 2 installation is incomplete" | `wsl --update` を実行して再起動 |
| "Virtualization support not detected" | 1.1 の BIOS 設定を確認 |
| `--gpus all` で `could not select device driver` | NVIDIA ドライバを最新化 → `wsl --shutdown` → Docker Desktop 再起動 |
| `nvidia-smi` は動くが CUDA が使えない | Docker Desktop を最新版に更新 |
| ディスク容量が逼迫 | `docker system prune` で不要なイメージを削除、4 の保存先変更 |
