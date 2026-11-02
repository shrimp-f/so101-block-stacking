"""SO-101 の MuJoCo モデルに関する定数とヘルパー。"""

from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101"
SCENE_XML = ASSET_DIR / "scene.xml"
# 描画用メッシュを間引いた版 (scripts/make_lowpoly.py で生成)。CPU 描画ではこちらを使う
LOWPOLY_ROBOT_XML = ASSET_DIR / "so101_lowpoly.xml"
LOWPOLY_SCENE_XML = ASSET_DIR / "scene_lowpoly.xml"

# LeRobot の SO-101 と同じ順序・名前
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = JOINT_NAMES[:5]

CONTROL_HZ = 30  # 実機の記録周期に合わせる


def load_model(xml_path: Path = SCENE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def joint_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])


def joint_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES])


def actuator_indices(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.actuator(name).id for name in JOINT_NAMES])


def joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    """(6, 2) の配列で各関節の [下限, 上限] (rad) を返す。"""
    return np.array([model.joint(name).range for name in JOINT_NAMES])


def step_control(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray) -> None:
    """関節の目標角 ctrl (rad) を与え、1 制御周期ぶんシミュレーションを進める。"""
    data.ctrl[actuator_indices(model)] = ctrl
    n_substeps = max(1, round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    mujoco.mj_step(model, data, nstep=n_substeps)


# --- LeRobot の単位との変換 ---------------------------------------------------
# 実機の LeRobot (use_degrees=True) は、腕の関節を「可動範囲の中央を 0 とする度数」、
# グリッパを「0 = 閉, 100 = 開」で扱う。シミュレーションのモデルはゼロ点が同じ (new calibration) なので、
# 腕は rad <-> deg の変換だけ、グリッパは可動範囲を 0〜100 に線形に写す。


def sim_to_lerobot(q: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    out = np.degrees(q).astype(np.float32)
    lo, hi = ranges[5]
    out[5] = (q[5] - lo) / (hi - lo) * 100.0
    return out


def lerobot_to_sim(a: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    q = np.radians(np.asarray(a, dtype=np.float64))
    lo, hi = ranges[5]
    q[5] = lo + a[5] / 100.0 * (hi - lo)
    return q


def lerobot_limits(ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LeRobot 単位での各関節の [下限], [上限]。"""
    low = np.degrees(ranges[:, 0]).astype(np.float32)
    high = np.degrees(ranges[:, 1]).astype(np.float32)
    low[5], high[5] = 0.0, 100.0
    return low, high
