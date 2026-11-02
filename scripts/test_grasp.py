"""Phase 2: 把持と 2 段積みの確認。

逆運動学で作った単純な軌道で、block0 を掴んで持ち上げ、block1 の上に置く。
接触パラメータ (指先パッド・摩擦・硬さ) が積み木タスクに使えるかを確かめる。
本格的なデモ生成用のポリシーは Phase 3 で作る。

使い方 (コンテナ内):
    python scripts/test_grasp.py [--seed 0] [--no-video]
出力:
    outputs/phase2/grasp_seed{N}.mp4   front | wrist カメラを並べた動画
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from so101_stack.controllers.ik import IKSolver, top_down_rotation
from so101_stack.envs.stack_env import StackEnv
from so101_stack.robot import ARM_JOINT_NAMES, joint_ranges, lerobot_to_sim, sim_to_lerobot

OUT_DIR = Path("outputs/phase2")
GRIPPER_OPEN = 40.0  # LeRobot 単位 (0 = 閉, 100 = 開)。指先の間隔 約 5.5cm
GRIPPER_CLOSE = 0.0
# 把持点 (指先パッドの中央) をブロック中心より上にして、指先 (把持点の約 1.9cm 下) がテーブルに当たらないようにする。
# 厚さ 1.5cm のブロックなら指先はテーブルから約 2mm 上、パッドはブロックの側面に約 1cm 接する
GRASP_Z_OFFSET = 0.0135
# 把持点は「固定ジョーにブロックが接した位置」なので、そのまま下ろすと固定ジョーがブロックの上に乗る。
# ジョーが開く方向に少しずらして下ろし、閉じる動作でブロックを固定ジョーへ押し付ける
GRASP_CLEARANCE = 0.008
# 真下向きで届く高さは約 10cm まで (scripts/reach_map.py) なので、上から近づく高さは低めにする
APPROACH_HEIGHT = 0.05
PLACE_APPROACH_HEIGHT = 0.03  # 2 段目を置く高さ + 5cm だと真下向きの上限を超える
# 逆運動学の初期値 (LeRobot 単位, 腕 5 関節)
IK_INIT_POSES = [[0, 0, 0, 90, 0], [0, -60, 60, 70, 0], [0, 30, 30, 60, 0], [0, -30, 90, 30, 0]]


def block_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def closing_axis(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


class Runner:
    def __init__(self, env: StackEnv, record: bool):
        self.env = env
        self.ik = IKSolver(env.model)
        self.ranges = joint_ranges(env.model)
        self.q_arm = env.data.qpos[self.ik.qadr].copy()
        self.inits = [lerobot_to_sim(np.append(p, 0.0), self.ranges)[:5] for p in IK_INIT_POSES]
        self.gripper = GRIPPER_OPEN
        self.frames = [] if record else None
        self.info = {}

    def choose_yaw(self, pos: np.ndarray, block_quat: np.ndarray, approach: float = APPROACH_HEIGHT) -> float:
        """ブロックの幅の方向を挟む 2 通りの向きのうち、手首の回転が最も小さいものを選ぶ。"""
        base = block_yaw(block_quat) + np.pi / 2  # ジョーが閉じる方向 = ブロックの長辺に垂直
        best = None
        for k in range(2):
            yaw = base + k * np.pi
            res = self.ik.solve_multi(pos + [0, 0, approach], top_down_rotation(yaw), [self.q_arm, *self.inits])
            roll = abs(res.q[ARM_JOINT_NAMES.index("wrist_roll")])
            if res.success and (best is None or roll < best[0]):
                best = (roll, yaw)
        if best is None:
            raise RuntimeError(f"IK failed for block at {pos}")
        return best[1]

    def move(self, target: np.ndarray, yaw: float, steps: int) -> None:
        """把持点を現在位置から target まで直線的に動かす。"""
        start = self.env.data.site("grasp_site").xpos.copy()
        for s in np.linspace(0, 1, steps)[1:]:
            res = self.ik.solve(start + s * (target - start), top_down_rotation(yaw), self.q_arm)
            self.q_arm = res.q
            self.step()
        actual = self.env.data.site("grasp_site").xpos
        print(
            f"  move to {np.round(target, 3)}: ik pos_err={res.pos_err * 1000:.1f}mm "
            f"rot_err={np.degrees(res.rot_err):.1f}deg, tracking err={np.linalg.norm(actual - target) * 1000:.1f}mm"
        )

    def set_gripper(self, value: float, steps: int = 15) -> None:
        self.gripper = value
        for _ in range(steps):
            self.step()

    def step(self) -> None:
        q = np.append(self.q_arm, 0.0)
        action = sim_to_lerobot(q, self.ranges)
        action[5] = self.gripper
        _, _, _, _, self.info = self.env.step(action)
        if self.frames is not None:
            self.frames.append(np.concatenate([self.env.render_camera("front"), self.env.render_camera("wrist")], axis=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = StackEnv(n_blocks=2, obs_type="state", max_episode_steps=10**6)
    env.reset(seed=args.seed)
    r = Runner(env, record=not args.no_video)
    pick, place = env.block_poses()
    h = env.cfg.block_dims[2]
    up = np.array([0, 0, APPROACH_HEIGHT])

    yaw = r.choose_yaw(pick[:3], pick[3:])
    pick_pos = pick[:3] + [0, 0, GRASP_Z_OFFSET] - GRASP_CLEARANCE * closing_axis(yaw)
    r.move(pick_pos + up, yaw, 40)
    r.move(pick_pos, yaw, 30)
    r.set_gripper(GRIPPER_CLOSE)
    r.move(pick_pos + up, yaw, 30)

    # 持ち上げた状態でのすべり: 把持点とブロック中心の高さの差
    lifted = env.block_poses()[0]
    site_z = env.data.site("grasp_site").xpos[2]
    print(f"lift: block z={lifted[2]:.4f} (from {pick[2]:.4f}), site z - block z = {site_z - lifted[2]:.4f} "
          f"(expected {GRASP_Z_OFFSET})")

    # 掴んだブロックは把持点にあるので、置くときはずらさない
    place_pos = place[:3] + [0, 0, h + GRASP_Z_OFFSET + 0.003]
    place_up = np.array([0, 0, PLACE_APPROACH_HEIGHT])
    yaw = r.choose_yaw(place_pos, place[3:], approach=PLACE_APPROACH_HEIGHT)
    r.move(place_pos + place_up, yaw, 45)
    r.move(place_pos, yaw, 30)
    r.set_gripper(GRIPPER_OPEN)
    r.move(place_pos + place_up, yaw, 30)
    for _ in range(30):
        r.step()
        if r.info["is_success"]:
            break

    print(f"result: {r.info}")
    if r.frames is not None:
        path = OUT_DIR / f"grasp_seed{args.seed}.mp4"
        imageio.mimwrite(path, r.frames, fps=30)
        print(f"saved: {path}")
    env.close()


if __name__ == "__main__":
    main()
