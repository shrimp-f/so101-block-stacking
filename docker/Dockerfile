# SO-101 積み木スタッキング シミュレータ環境
# - Python 3.12 (lerobot 0.6 系の要件)
# - PyTorch は cu126 ビルド (GTX 1070 = Pascal sm_61 をサポートし、ホストドライバの CUDA 12.6 に適合)
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ARG LEROBOT_VERSION=0.6.1
ARG TORCH_VERSION=2.11.*
ARG TORCHVISION_VERSION=0.26.*
ARG MUJOCO_VERSION=3.17.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # コンテナ内で OpenGL/EGL を使えるようにする
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# システムパッケージ
# - ffmpeg: LeRobotDataset の動画エンコード/デコード
# - libegl1/libgl1/libosmesa6: MuJoCo のオフスクリーン描画 (EGL / OSMesa)
# - xvfb/x11vnc/novnc: ブラウザから MuJoCo viewer を操作するための仮想ディスプレイ
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential git curl ca-certificates \
        ffmpeg \
        libegl1 libgl1 libgles2 libosmesa6 libglfw3 \
        xvfb x11vnc novnc websockify \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv $VIRTUAL_ENV && pip install --upgrade pip

# PyTorch を先に入れ、lerobot のインストールで別ビルドに置き換わらないようにする
RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu126

RUN pip install \
        "lerobot[training,dataset,diffusion,viz]==${LEROBOT_VERSION}" \
        "mujoco==${MUJOCO_VERSION}" \
        "gymnasium>=1.1" \
        imageio[ffmpeg]

# メッシュ加工 (描画用の低ポリゴンメッシュ生成)
RUN pip install trimesh fast-simplification

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
