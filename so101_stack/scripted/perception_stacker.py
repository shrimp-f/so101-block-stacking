"""AI の認識結果だけを使って積み上げるポリシー。

ScriptedStacker と同じ動作手順で動かすが、ブロックの位置・向きはシミュレーション内部の正解ではなく、
BlockDetector (OWLv2 による物体検出 + 深度) で毎回カメラ画像から推定する。
掴めたかどうかも、ブロックの正解の位置ではなくグリッパの開き具合で判断する。
置く位置は、タワーの位置と何段目かから計算する (積んだブロックは見ない)。
"""

from collections.abc import Iterator

import mujoco
import numpy as np

from so101_stack.envs.stack_env import REST_POSE, TOWER_POS, StackEnv
from so101_stack.perception.block_detector import BlockDetector, Detection
from so101_stack.robot import joint_qpos_indices, lerobot_to_sim, sim_to_lerobot
from so101_stack.scripted.stack_policy import FINAL_HOLD_STEPS, GRIPPER_OPEN, MAX_GRASP_TRIES, ScriptedStacker

TOWER_EXCLUDE_R = 0.04  # タワー位置からこの距離以内の検出は、積んだブロックとみなして掴む対象にしない
HOLD_MIN_GRIPPER = 5.0  # 閉じた後のグリッパの値 (LeRobot 単位) がこれより大きければ、ブロックを挟んでいる


class PerceptionStacker(ScriptedStacker):
    def __init__(self, env: StackEnv, detector: BlockDetector):
        super().__init__(env)
        self.detector = detector
        self.qidx = joint_qpos_indices(env.model)
        # (何ステップ目か, 検出結果, 検出に使った画像)。動画用
        self.events: list[tuple[int, list[Detection], np.ndarray]] = []
        self._n_actions = 0

    def act(self) -> np.ndarray | None:
        action = super().act()
        if action is not None:
            self._n_actions += 1
        return action

    def _choose_block(self) -> np.ndarray | None:
        """カメラ画像からブロックを検出し、タワーに最も近いもの (積んだものを除く) の姿勢を返す。"""
        found = self.detector.detect(self.env.data)
        self.events.append((self._n_actions, found, self.detector.last_rgb))
        candidates = [d for d in found if np.linalg.norm(d.xy - TOWER_POS) > TOWER_EXCLUDE_R]
        if not candidates:
            return None
        d = min(candidates, key=lambda d: np.linalg.norm(d.xy - TOWER_POS))
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), d.yaw)
        return np.concatenate([[*d.xy, self.block_dims[2] / 2], quat])

    def gripper_value(self) -> float:
        return float(sim_to_lerobot(self.env.data.qpos[self.qidx], self.ranges)[5])

    def _run(self) -> Iterator[np.ndarray]:
        h = self.block_dims[2]
        tower_yaw = np.arctan2(TOWER_POS[1], TOWER_POS[0])
        rest = lerobot_to_sim(REST_POSE, self.ranges)[:5]
        for level in range(1, self.env.cfg.n_blocks + 1):
            for _ in range(MAX_GRASP_TRIES):
                # 腕がカメラの視界を遮らないよう、待機姿勢に戻ってから検出する
                if self._n_actions > 0:
                    yield from self._move_joints(rest, gripper=GRIPPER_OPEN)
                pose = self._choose_block()
                if pose is None:
                    return  # 掴むブロックが見つからない
                yield from self._pick_at(pose)
                if self.gripper_value() > HOLD_MIN_GRIPPER:
                    break
                yield from self._set_gripper(GRIPPER_OPEN)
            else:
                return
            yield from self._place_at(np.array([*TOWER_POS, h / 2 + (level - 1) * h]), tower_yaw, level)
        yield from self._move_joints(lerobot_to_sim(REST_POSE, self.ranges)[:5], gripper=REST_POSE[5])
        for _ in range(FINAL_HOLD_STEPS):
            yield self._action()
