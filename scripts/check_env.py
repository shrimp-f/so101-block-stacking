"""Phase 0 の完了条件を確認する: torch が GPU を認識し、MuJoCo でオフスクリーン描画できること。

使い方 (コンテナ内):
    python scripts/check_env.py
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

import lerobot

OUT_DIR = Path("outputs")

SCENE = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="0.5 0.5 0.01" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.1"><freejoint/><geom type="box" size="0.0125 0.0125 0.0125" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.01 0 0.2"><freejoint/><geom type="box" size="0.0125 0.0125 0.0125" rgba="0.2 0.4 0.9 1"/></body>
    <camera name="front" pos="0 -0.25 0.12" xyaxes="1 0 0 0 0.4 1"/>
  </worldbody>
</mujoco>
"""


def check_torch() -> None:
    print(f"torch {torch.__version__} / CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    print(f"  device: {torch.cuda.get_device_name(0)}, capability: {torch.cuda.get_device_capability(0)}")
    print(f"  arch list: {torch.cuda.get_arch_list()}")
    x = torch.randn(1024, 1024, device="cuda")
    print(f"  matmul on GPU OK: {(x @ x).sum().item():.1f}")


def check_mujoco() -> None:
    print(f"mujoco {mujoco.__version__} / MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    for _ in range(500):  # ブロックを落下させて積み重ねる
        mujoco.mj_step(model, data)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="front")
        img = renderer.render()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "check_render.png"
    Image.fromarray(img).save(path)
    print(f"  rendered {img.shape}, mean={np.mean(img):.1f} -> {path}")


if __name__ == "__main__":
    print(f"lerobot {lerobot.__version__}")
    check_torch()
    check_mujoco()
