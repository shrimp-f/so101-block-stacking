"""上からの把持で把持点が届く範囲を調べる。

ベースからの水平距離 r と高さ z の格子で逆運動学を解き、届くかどうかを表示する。
ブロックを置く範囲や積み上げられる段数を決めるのに使う。

- 真下向き: ジョーの閉じる方向は 2 通り (半径方向・接線方向) を試す
- 傾けた姿勢: ジョーの閉じる方向を接線方向 (水平) に固定し、指を前後に傾ける

使い方 (コンテナ内):
    python scripts/reach_map.py
"""

import numpy as np

from so101_stack.controllers.ik import IKSolver, tilted_grasp_rotation, top_down_rotation
from so101_stack.envs.scene import SceneConfig, build_model
from so101_stack.robot import joint_ranges, lerobot_to_sim

# 逆運動学の初期値 (LeRobot 単位, 腕 5 関節)。局所解に落ちるのを避けるため複数から解く
INIT_POSES = [
    [0, 0, 0, 90, 0],
    [0, -60, 60, 70, 0],
    [0, 30, 30, 60, 0],
    [0, -30, 90, 30, 0],
    [0, -60, 30, 90, 0],
]
TILTS_DEG = range(-60, 61, 10)


def reachable(ik: IKSolver, inits, pos: np.ndarray, rotations) -> bool:
    return any(ik.solve(pos, rot, q0).success for rot in rotations for q0 in inits)


def main() -> None:
    model = build_model(SceneConfig())
    ik = IKSolver(model)
    ranges = joint_ranges(model)
    inits = [lerobot_to_sim(np.append(p, 0.0), ranges)[:5] for p in INIT_POSES]

    rs = np.arange(0.08, 0.42, 0.02)
    zs = np.arange(0.30, -0.001, -0.02)
    azimuth = 0.0
    tangent_yaw = azimuth + np.pi / 2
    print("o = 真下向きで届く, t = 傾ければ届く (数字は必要な最小の傾き [10deg 単位]), . = 届かない")
    print("rows: z [m], cols: r [m]")
    print("        " + "".join(f"{r:5.2f}" for r in rs))
    for z in zs:
        row = []
        for r in rs:
            pos = np.array([r * np.cos(azimuth), r * np.sin(azimuth), z])
            if reachable(ik, inits, pos, [top_down_rotation(azimuth), top_down_rotation(tangent_yaw)]):
                row.append("    o")
                continue
            # 小さい傾きから順に試す
            mark = "    ."
            for tilt in sorted(TILTS_DEG, key=abs):
                if tilt and reachable(ik, inits, pos, [tilted_grasp_rotation(tangent_yaw, np.radians(tilt))]):
                    mark = f"  t{tilt // 10:+d}"
                    break
            row.append(mark)
        print(f"z={z:5.2f} " + "".join(row))


if __name__ == "__main__":
    main()
