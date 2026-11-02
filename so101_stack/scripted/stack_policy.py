"""デモ生成用のスクリプトポリシー: ジェンガのブロックを TOWER_POS に 1 列に積み上げる。

シミュレーションの内部状態 (ブロックの位置姿勢) を直接使う「答えを知っている」ポリシー。
各動作の開始時にブロックの位置を読み直し、逆運動学で把持点の直線軌道を作って関節の目標角を出す。

1 個あたりの流れ:
    近づく → 下ろす → 掴む → 持ち上げる (掴めていなければやり直す)
    → タワーの横 → タワーの上 → 下ろす → 放す → 上に逃げる

制約:
- 把持点の姿勢は真下向きのみ。届く高さの都合で、ジェンガのブロックで 4〜5 段目まで
  (doc/plan.md の Phase 2「届く範囲」を参照)
"""

from collections.abc import Iterator

import numpy as np

from so101_stack.controllers.ik import IKSolver, top_down_rotation
from so101_stack.envs.stack_env import REST_POSE, TOWER_POS, StackEnv
from so101_stack.robot import ARM_JOINT_NAMES, CONTROL_HZ, joint_ranges, lerobot_to_sim, sim_to_lerobot

GRIPPER_OPEN = 40.0  # LeRobot 単位 (0 = 閉, 100 = 開)。指先の間隔 約 5.5cm
GRIPPER_CLOSE = 0.0
# 把持点 (指先パッドの中央) はブロック中心の 1.35cm 上。指先がテーブルの約 2mm 上になる
GRASP_Z_OFFSET = 0.0135
# 下ろすときは把持点をジョーが開く方向にずらす (ずらさないと固定ジョーがブロックに乗り上げる)
GRASP_CLEARANCE = 0.008
PICK_APPROACHES = (0.05, 0.03)  # 掴む前に真上で止まる高さ。届かなければ低い方を試す
PLACE_DROP = 0.002  # 置くときにブロックを放す高さ
PLACE_CLEARANCE = 0.025  # タワーの上に運ぶときの、置く位置からの高さ
SIDE_OFFSET = 0.06  # タワーの真上に入る前に寄る位置の、タワーからの水平距離
LIFT_MIN_Z = 0.06  # 持ち上げたときの把持点の最低の高さ (テーブル上のブロックに当たらない)
FAST_SPEED = 0.20  # [m/s]
SLOW_SPEED = 0.06  # [m/s] 掴む・置くときの上下動
MAX_GRASP_TRIES = 3
FINAL_HOLD_STEPS = 60

# 逆運動学の初期値 (LeRobot 単位, 腕 5 関節)。局所解に落ちるのを避けるため複数から解く
IK_INIT_POSES = [[0, 0, 0, 90, 0], [0, -60, 60, 70, 0], [0, 30, 30, 60, 0], [0, -30, 90, 30, 0]]
ROLL = ARM_JOINT_NAMES.index("wrist_roll")


class PlanningError(RuntimeError):
    """逆運動学が解けず、動作を作れなかった。"""


def block_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def closing_axis(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScriptedStacker:
    def __init__(self, env: StackEnv):
        self.env = env
        self.ik = IKSolver(env.model)
        self.ranges = joint_ranges(env.model)
        self.inits = [lerobot_to_sim(np.append(p, 0.0), self.ranges)[:5] for p in IK_INIT_POSES]
        self.block_dims = np.array(env.cfg.block_dims)
        self._actions: Iterator[np.ndarray] | None = None

    def reset(self) -> None:
        d = self.env.data
        self.q_arm = d.qpos[self.ik.qadr].copy()
        self.gripper = float(REST_POSE[5])
        self.yaw = 0.0
        self._actions = self._run()

    def act(self) -> np.ndarray | None:
        """次の制御周期の行動 (LeRobot 単位)。動作が終わったら None。"""
        return next(self._actions, None)

    # --- 全体の流れ ----------------------------------------------------------

    def _run(self) -> Iterator[np.ndarray]:
        order = self._pick_order()
        for i in order:
            for _ in range(MAX_GRASP_TRIES):
                yield from self._pick(i)
                if self._is_grasped(i):
                    break
                yield from self._set_gripper(GRIPPER_OPEN)
            else:
                return  # 掴めなかった。エピソードは失敗として捨てる
            yield from self._place(i)
        yield from self._move_joints(lerobot_to_sim(REST_POSE, self.ranges)[:5], gripper=REST_POSE[5])
        for _ in range(FINAL_HOLD_STEPS):
            yield self._action()

    def _pick_order(self) -> list[int]:
        """タワーに近いブロックから運ぶ。"""
        pos = self.env.block_poses()[:, :2]
        return list(np.argsort(np.linalg.norm(pos - TOWER_POS, axis=1)))

    # --- 掴む・置く -----------------------------------------------------------

    def _pick(self, i: int) -> Iterator[np.ndarray]:
        pose = self.env.block_poses()[i]
        center = pose[:3] + [0, 0, GRASP_Z_OFFSET]

        # ブロックの幅の方向を挟む 2 通りのうち、近づく位置・掴む位置の両方に届き、手首の回転が小さい方
        candidates = []
        for approach in PICK_APPROACHES:
            for yaw in block_yaw(pose[3:]) + np.pi / 2 + np.array([0, np.pi]):
                target = center - GRASP_CLEARANCE * closing_axis(yaw)
                above = self._solve(target + [0, 0, approach], yaw)
                if above is not None and self._solve(target, yaw, q_init=above) is not None:
                    candidates.append((abs(above[ROLL]), yaw, target, above, approach))
            if candidates:
                break
        if not candidates:
            raise PlanningError(f"cannot reach block {i} at {np.round(pose[:3], 3)}")
        _, yaw, target, above, approach = min(candidates, key=lambda c: c[0])

        yield from self._move_joints(above, gripper=GRIPPER_OPEN, yaw=yaw)
        yield from self._move_line(target, SLOW_SPEED)
        yield from self._set_gripper(GRIPPER_CLOSE)
        lift_z = max(LIFT_MIN_Z, target[2] + approach)
        yield from self._move_line(np.append(target[:2], lift_z), SLOW_SPEED)

    def _place(self, i: int) -> Iterator[np.ndarray]:
        h = self.block_dims[2]
        chain = self._tower_chain(exclude=i)
        if chain:
            top = self.env.block_poses()[chain[-1]]
            place_center = top[:3] + [0, 0, h]
            base_yaw = block_yaw(top[3:])
        else:
            # 1 段目: 長辺を半径方向 (ロボットから見て前後) に向けて置く
            place_center = np.array([*TOWER_POS, h / 2])
            base_yaw = np.arctan2(TOWER_POS[1], TOWER_POS[0])
        target = place_center + [0, 0, GRASP_Z_OFFSET + PLACE_DROP]
        above = target + [0, 0, PLACE_CLEARANCE]

        # タワーの真上に入る前に、今いる側に SIDE_OFFSET だけ離れた位置に寄る (掴んだブロックがタワーに当たらないように)
        here = self.env.data.site("grasp_site").xpos.copy()
        side_dir = here[:2] - place_center[:2]
        side_dir /= np.linalg.norm(side_dir) + 1e-9
        side = np.append(place_center[:2] + SIDE_OFFSET * side_dir, above[2])

        candidates = []
        for yaw in base_yaw + np.pi / 2 + np.array([0, np.pi]):
            q_side = self._solve(side, yaw)
            if q_side is None:
                continue
            q_above = self._solve(above, yaw, q_init=q_side)
            if q_above is not None and self._solve(target, yaw, q_init=q_above) is not None:
                candidates.append((abs(q_side[ROLL]), yaw))
        if not candidates:
            raise PlanningError(f"cannot reach tower level {len(chain) + 1} at {np.round(target, 3)}")
        _, yaw = min(candidates, key=lambda c: c[0])

        yield from self._move_line(side, FAST_SPEED, yaw=yaw)
        yield from self._move_line(above, SLOW_SPEED)
        yield from self._move_line(target, SLOW_SPEED)
        yield from self._set_gripper(GRIPPER_OPEN)
        yield from self._move_line(above, SLOW_SPEED)

    def _tower_chain(self, exclude: int) -> list[int]:
        """TOWER_POS に立っているブロックの列 (下から順)。運んでいるブロックは除く。"""
        w = self.block_dims[1]
        for chain in self.env._stack_chains():
            chain = [b for b in chain if b != exclude]
            if chain and np.linalg.norm(self.env.block_poses()[chain[0], :2] - TOWER_POS) < w:
                return chain
        return []

    def _is_grasped(self, i: int) -> bool:
        block = self.env.block_poses()[i, :3]
        site = self.env.data.site("grasp_site").xpos
        return block[2] > self.block_dims[2] and np.linalg.norm(block[:2] - site[:2]) < 0.02

    # --- 動作の生成 -----------------------------------------------------------

    def _solve(self, pos: np.ndarray, yaw: float, q_init: np.ndarray | None = None) -> np.ndarray | None:
        q0 = self.q_arm if q_init is None else q_init
        res = self.ik.solve_multi(pos, top_down_rotation(yaw), [q0, *self.inits])
        return res.q if res.success else None

    def _move_line(self, target: np.ndarray, speed: float, yaw: float | None = None) -> Iterator[np.ndarray]:
        """把持点を今の位置から target まで直線で動かす。yaw を渡すと向きも途中で回す。"""
        start = self.env.data.site("grasp_site").xpos.copy()
        yaw0 = self.yaw
        yaw1 = self.yaw if yaw is None else yaw0 + wrap_angle(yaw - yaw0)
        steps = max(8, int(np.linalg.norm(target - start) / speed * CONTROL_HZ))
        for s in np.linspace(0, 1, steps + 1)[1:]:
            self.yaw = yaw0 + s * (yaw1 - yaw0)
            res = self.ik.solve(start + s * (target - start), top_down_rotation(self.yaw), self.q_arm)
            self.q_arm = res.q
            yield self._action()

    def _move_joints(
        self, q_target: np.ndarray, gripper: float | None = None, yaw: float | None = None, speed: float = 1.5
    ) -> Iterator[np.ndarray]:
        """関節空間で q_target まで補間する (姿勢が大きく変わる移動用)。speed は最も動く関節の角速度 [rad/s]。"""
        if gripper is not None:
            self.gripper = gripper
        q0 = self.q_arm.copy()
        steps = max(10, int(np.max(np.abs(q_target - q0)) / speed * CONTROL_HZ))
        for s in np.linspace(0, 1, steps + 1)[1:]:
            self.q_arm = q0 + s * (q_target - q0)
            yield self._action()
        if yaw is not None:
            self.yaw = yaw

    def _set_gripper(self, value: float, steps: int = 12) -> Iterator[np.ndarray]:
        self.gripper = value
        for _ in range(steps):
            yield self._action()

    def _action(self) -> np.ndarray:
        action = sim_to_lerobot(np.append(self.q_arm, 0.0), self.ranges)
        action[5] = self.gripper
        return action
