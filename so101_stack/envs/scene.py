"""積み木スタッキングのシーン (SO-101 + テーブル + ブロック + カメラ) を MjSpec で組み立てる。"""

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_stack.robot import LOWPOLY_ROBOT_XML

# 衝突フィルタのビット。2 つの geom は (a.contype & b.conaffinity) or (b.contype & a.conaffinity) のとき接触する
COL_WORLD = 1  # テーブル・アームのメッシュ
COL_GRIPPER_MESH = 2  # グリッパのジョーのメッシュ (凸包が把持の邪魔をするのでブロックとは接触させない)
COL_BLOCK = 4
COL_PAD = 8  # 指先に追加する箱形の当たり判定

# 凸包が指の間を埋めてしまうメッシュ。ブロックとの接触は指先パッドに任せる
GRIPPER_MESHES = ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1")

# ジェンガのブロック風の木目の色 (ブロックごとに少し変える)
BLOCK_COLORS = [
    (0.87, 0.72, 0.50, 1),
    (0.82, 0.66, 0.44, 1),
    (0.90, 0.77, 0.56, 1),
    (0.78, 0.62, 0.40, 1),
    (0.85, 0.70, 0.47, 1),
]
TABLE_RGBA = (0.45, 0.47, 0.50, 1)  # ブロックと見分けやすいグレー

PAD_FRICTION = (1.5, 0.005, 0.0001)
# MuJoCo は接触する 2 つの geom の摩擦係数の大きい方を使う。木どうし 0.5、木とテーブル 0.6
BLOCK_FRICTION = (0.5, 0.005, 0.0001)
TABLE_FRICTION = (0.6, 0.005, 0.0001)
# 既定の接触 (solref 0.02) はやわらかく、グリッパを押し付けると 2cm 以上めり込むので硬くする
CONTACT_SOLREF = (0.004, 1.0)
CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

# 固定ジョーの内面の位置 (グリッパ座標系の x)。把持点はここにブロックの側面が接したときのブロック中心
FIXED_PAD_FACE_X = -0.008
GRASP_SITE_Z = -0.086  # 指先パッドの中央の高さ
# 公式モデルの gripperframe と同じ向き: x 軸 = 指が伸びる方向, z 軸 = ジョーが閉じる方向
GRASP_SITE_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)

# 公式モデル (so101_new_calib_camera.xml) のリストカメラの位置。グリッパ座標系
WRIST_CAMERA_POS = np.array([0.0025, 0.0734, 0.0066])


@dataclass
class SceneConfig:
    n_blocks: int = 3
    # ジェンガのブロック: 長さ (x) × 幅 (y) × 厚さ (z) [m]。平らに寝かせて積み、幅の方向を挟んで掴む
    block_dims: tuple[float, float, float] = (0.075, 0.025, 0.015)
    block_mass: float = 0.015  # [kg]
    render_width: int = 640
    render_height: int = 480

    @property
    def grasp_site_pos(self) -> tuple[float, float, float]:
        return (FIXED_PAD_FACE_X + self.block_dims[1] / 2, 0.0, GRASP_SITE_Z)


def look_at_quat(pos, target, right_hint) -> np.ndarray:
    """pos から target を見るカメラの姿勢 (MuJoCo のカメラは -z 方向を向き、+y が画像の上)。

    right_hint は画像の右方向にしたいおおよその向き。
    """
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.asarray(right_hint, float) - np.dot(right_hint, f) * f
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -f]).flatten())
    return quat


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(LOWPOLY_ROBOT_XML))
    spec.modelname = "so101_stack"

    # 物理設定: 30Hz 制御でちょうど 20 サブステップ。把持のすべりを抑えるため楕円摩擦錐 + 高い impratio
    spec.option.timestep = 1.0 / 600
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    # CPU 描画向けの設定 (doc/plan.md Phase 1 の結果メモを参照)
    spec.visual.quality.shadowsize = 2048
    spec.visual.quality.offsamples = 0
    spec.visual.global_.offwidth = max(cfg.render_width, 640)
    spec.visual.global_.offheight = max(cfg.render_height, 480)

    _add_world(spec)
    _add_gripper_parts(spec, cfg)
    _add_blocks(spec, cfg)
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    return build_spec(cfg).compile()


def _add_world(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    wb.add_light(name="top", pos=[0.25, -0.2, 1.0], dir=[-0.15, 0.2, -1.0], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.6, 0.6, 0.01],
        pos=[0.15, 0, 0],
        rgba=TABLE_RGBA,
        friction=TABLE_FRICTION,
        contype=COL_WORLD,
        conaffinity=COL_WORLD,
        solref=CONTACT_SOLREF,
        solimp=CONTACT_SOLIMP,
    )
    # ロボットの右前方の斜め上から、作業領域と積み上げの高さが見えるように置く固定カメラ
    front_pos = [0.50, -0.38, 0.32]
    wb.add_camera(
        name="front", pos=front_pos, quat=look_at_quat(front_pos, [0.17, 0.0, 0.06], [0.75, 0.66, 0]), fovy=50
    )
    # SmolVLA の事前学習データ (実機 SO-100) によくある配置に近づけた、真上から見下ろすカメラと横からのカメラ。
    # 学習なしの SmolVLA を試すときだけ使う (camera1 = top, camera2 = wrist, camera3 = side)
    top_pos = [0.20, 0.0, 0.75]
    wb.add_camera(name="top", pos=top_pos, quat=look_at_quat(top_pos, [0.20, 0.0, 0.0], [0, -1, 0]), fovy=50)
    side_pos = [0.22, -0.55, 0.22]
    wb.add_camera(name="side", pos=side_pos, quat=look_at_quat(side_pos, [0.20, 0.0, 0.05], [1, 0, 0]), fovy=50)


def _add_gripper_parts(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    for geom in spec.geoms:
        if geom.meshname in GRIPPER_MESHES and geom.contype != 0:
            geom.contype = COL_GRIPPER_MESH
            geom.conaffinity = COL_WORLD

    pad = dict(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        contype=COL_PAD,
        conaffinity=COL_WORLD | COL_BLOCK,
        friction=PAD_FRICTION,
        condim=4,
        solref=CONTACT_SOLREF,
        solimp=CONTACT_SOLIMP,
        mass=0.001,
        rgba=[0.15, 0.15, 0.15, 1],
        group=1,
    )
    gripper = spec.body("gripper")
    # 固定ジョーの内面に沿った厚さ 3mm のパッド。指先から 3.2cm
    gripper.add_geom(
        name="fixed_pad", size=[0.0015, 0.007, 0.016], pos=[FIXED_PAD_FACE_X - 0.0015, 0, GRASP_SITE_Z], **pad
    )
    # 可動ジョーの内面。ジョーの座標系では指が -y 方向に伸び、内面は x = -0.0123
    spec.body("moving_jaw_so101_v1").add_geom(
        name="moving_pad", size=[0.0015, 0.016, 0.007], pos=[-0.0108, -0.0626, 0.0188], **pad
    )
    gripper.add_site(name="grasp_site", pos=cfg.grasp_site_pos, quat=GRASP_SITE_QUAT, size=[0.004, 0, 0], group=4)

    # リストカメラ: 把持点の少し先を見る。画像の右方向 = ジョーが閉じる方向
    look_target = np.array(cfg.grasp_site_pos) + [0, 0, -0.02]
    gripper.add_camera(
        name="wrist", pos=WRIST_CAMERA_POS, quat=look_at_quat(WRIST_CAMERA_POS, look_target, [1, 0, 0]), fovy=75
    )


def _add_blocks(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    half = np.array(cfg.block_dims) / 2
    for i in range(cfg.n_blocks):
        body = spec.worldbody.add_body(name=f"block{i}", pos=[0.2, 0.05 * i, half[2]])
        body.add_freejoint(name=f"block{i}_joint")
        body.add_geom(
            name=f"block{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            mass=cfg.block_mass,
            rgba=BLOCK_COLORS[i % len(BLOCK_COLORS)],
            friction=BLOCK_FRICTION,
            condim=4,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            contype=COL_BLOCK,
            conaffinity=COL_WORLD | COL_BLOCK | COL_PAD,
        )
