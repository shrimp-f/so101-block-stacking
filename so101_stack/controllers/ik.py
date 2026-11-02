"""減衰最小二乗法 (DLS) による SO-101 の逆運動学。

把持点 (grasp_site) の位置と姿勢を目標に、腕の 5 関節の角度を求める。
グリッパの関節は動かさない。
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_stack.robot import ARM_JOINT_NAMES


@dataclass
class IKResult:
    q: np.ndarray  # 腕 5 関節の角度 [rad]
    pos_err: float  # [m]
    rot_err: float  # [rad]
    success: bool


def top_down_rotation(yaw: float) -> np.ndarray:
    """指を真下に向け、ジョーの閉じる方向をワールドの yaw 方向にそろえた把持点の姿勢 (回転行列)。

    把持点の座標系は x 軸 = 指の方向, z 軸 = ジョーが閉じる方向。
    """
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def tilted_grasp_rotation(yaw: float, tilt: float) -> np.ndarray:
    """top_down_rotation を、ジョーが閉じる方向 (水平) を軸に tilt [rad] だけ傾けた姿勢。

    閉じる方向が水平のままなのでパッドの面は垂直に保たれ、ブロックの側面をまっすぐ挟める。
    真下向きでは届かない高い位置 (積み上げの上段) に置くときに使う。
    """
    R = top_down_rotation(yaw)
    axis = R[:, 2]
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, tilt)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    return rot.reshape(3, 3) @ R


class IKSolver:
    def __init__(self, model: mujoco.MjModel, site: str = "grasp_site"):
        self.model = model
        self.data = mujoco.MjData(model)  # シミュレーション本体の状態を壊さないよう専用の MjData を使う
        self.site_id = model.site(site).id
        self.qadr = np.array([model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES])
        self.dadr = np.array([model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES])
        self.ranges = np.array([model.joint(n).range for n in ARM_JOINT_NAMES])

    def solve(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        q_init: np.ndarray,
        rot_weight: float = 0.3,
        damping: float = 1e-3,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = np.radians(2.0),
    ) -> IKResult:
        d = self.data
        q = np.clip(np.array(q_init, dtype=float), self.ranges[:, 0], self.ranges[:, 1])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        quat_err = np.zeros(4)
        site_quat = np.zeros(4)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rot.flatten())

        for _ in range(max_iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            e_pos = pos - d.site_xpos[self.site_id]
            mujoco.mju_mat2Quat(site_quat, d.site_xmat[self.site_id])
            # 現在姿勢から目標姿勢への回転 (ワールド座標系の回転ベクトル)
            mujoco.mju_negQuat(site_quat, site_quat)
            mujoco.mju_mulQuat(quat_err, target_quat, site_quat)
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat_err, 1.0)

            pos_err, rot_err = np.linalg.norm(e_pos), np.linalg.norm(e_rot)
            if pos_err < pos_tol and rot_err < rot_tol:
                break

            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            e = np.concatenate([e_pos, rot_weight * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + dq, self.ranges[:, 0], self.ranges[:, 1])

        return IKResult(q=q, pos_err=pos_err, rot_err=rot_err, success=pos_err < pos_tol * 2 and rot_err < rot_tol * 2)

    def solve_multi(self, pos: np.ndarray, rot: np.ndarray, inits: list[np.ndarray], **kwargs) -> IKResult:
        """複数の初期値から解き、成功した解のうち最初の初期値に最も近いものを返す。

        初期値によって局所解に落ちることがあるので、届く範囲の境目付近ではこちらを使う。
        """
        results = [self.solve(pos, rot, q0, **kwargs) for q0 in inits]
        ok = [r for r in results if r.success]
        if not ok:
            return min(results, key=lambda r: r.pos_err)
        return min(ok, key=lambda r: np.linalg.norm(r.q - inits[0]))
