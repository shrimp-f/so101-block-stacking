"""MuJoCo の描画速度を測る。描画バックエンドや解像度の比較用。

使い方 (コンテナ内):
    python scripts/bench_render.py [--width 640 --height 480 --frames 60 --xml assets/so101/scene_lowpoly.xml]
"""

import argparse
import time
from pathlib import Path

import mujoco
from OpenGL import GL

from so101_stack.robot import SCENE_XML, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--xml", type=Path, default=SCENE_XML)
    args = parser.parse_args()

    model, data = load_model(args.xml)
    print(f"{args.xml.name}: {model.nmeshface} mesh faces")
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        print(f"GL_RENDERER: {GL.glGetString(GL.GL_RENDERER).decode()}")
        renderer.update_scene(data)
        renderer.render()  # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(args.frames):
            renderer.update_scene(data)
            renderer.render()
        fps = args.frames / (time.perf_counter() - t0)
    print(f"{args.width}x{args.height}: {fps:.1f} fps")


if __name__ == "__main__":
    main()
