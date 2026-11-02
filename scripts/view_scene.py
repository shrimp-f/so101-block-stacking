"""Phase 1: SO-101 の各関節を順番に動かし、動画と追従誤差を出力する。

使い方 (コンテナ内):
    python scripts/view_scene.py
出力:
    outputs/phase1/joint_sweep.mp4   関節スイープの動画
    outputs/phase1/home.png          全関節 0 rad の姿勢
"""

import time
from pathlib import Path

import imageio
import mujoco
import numpy as np

from so101_stack.robot import (
    CONTROL_HZ,
    JOINT_NAMES,
    LOWPOLY_SCENE_XML,
    joint_qpos_indices,
    joint_ranges,
    load_model,
    step_control,
)

OUT_DIR = Path("outputs/phase1")
WIDTH, HEIGHT = 640, 480
SWEEP_SEC = 2.0  # 1 区間 (中央→下限 など) にかける時間
HOLD_SEC = 0.5  # 区間の終わりで静止させる時間 (定常誤差を測る)
MARGIN = 0.05  # 関節リミットぎりぎりまで動かさないための余裕 (rad)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.15]
    cam.distance = 0.75
    cam.azimuth = 135
    cam.elevation = -20
    return cam


def build_trajectory(ranges: np.ndarray) -> list[tuple[str, np.ndarray, bool]]:
    """関節ごとに 0 → 下限 → 上限 → 0 と動かす目標角の列を作る。

    各要素は (関節名, 目標角, 静止区間の最終フレームか)。
    """
    traj = []
    n_move = int(SWEEP_SEC * CONTROL_HZ)
    n_hold = int(HOLD_SEC * CONTROL_HZ)
    for j, name in enumerate(JOINT_NAMES):
        lo, hi = ranges[j] + [MARGIN, -MARGIN]
        for start, end in [(0.0, lo), (lo, hi), (hi, 0.0)]:
            for s in np.linspace(start, end, n_move):
                target = np.zeros(len(JOINT_NAMES))
                target[j] = s
                traj.append((name, target, False))
            traj += [(name, target, False)] * (n_hold - 1) + [(name, target, True)]
    return traj


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, data = load_model(LOWPOLY_SCENE_XML)
    qidx = joint_qpos_indices(model)
    ranges = joint_ranges(model)

    print(f"timestep={model.opt.timestep}s, control={CONTROL_HZ}Hz")
    for name, (lo, hi) in zip(JOINT_NAMES, ranges):
        print(f"  {name:14s} range [{np.degrees(lo):7.1f}, {np.degrees(hi):7.1f}] deg")

    cam = make_camera()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # 全関節 0 rad で静止させた姿勢
    for _ in range(CONTROL_HZ):
        step_control(model, data, np.zeros(len(JOINT_NAMES)))
    renderer.update_scene(data, camera=cam)
    imageio.imwrite(OUT_DIR / "home.png", renderer.render())

    traj = build_trajectory(ranges)
    errors = {name: [] for name in JOINT_NAMES}
    render_time = 0.0
    with imageio.get_writer(OUT_DIR / "joint_sweep.mp4", fps=CONTROL_HZ) as writer:
        for name, target, is_hold_end in traj:
            step_control(model, data, target)
            if is_hold_end:  # 静止区間の最後のフレームで定常誤差を記録
                errors[name].append(np.degrees(np.abs(data.qpos[qidx] - target)))
            t0 = time.perf_counter()
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            render_time += time.perf_counter() - t0

    print(f"frames={len(traj)}, render {len(traj) / render_time:.1f} fps ({WIDTH}x{HEIGHT})")
    print("steady-state error [deg] at hold points (lower limit / upper limit / zero):")
    for name in JOINT_NAMES:
        # 動かしている関節の誤差と、他の関節の誤差の最大値
        j = JOINT_NAMES.index(name)
        own = [f"{e[j]:5.2f}" for e in errors[name]]
        others = max(np.delete(e, j).max() for e in errors[name])
        print(f"  {name:14s} {' / '.join(own)}   (others max {others:.2f})")
    print(f"saved: {OUT_DIR / 'joint_sweep.mp4'}, {OUT_DIR / 'home.png'}")


if __name__ == "__main__":
    main()
