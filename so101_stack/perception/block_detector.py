"""AI (オープン語彙の物体検出モデル OWLv2) でカメラ画像からブロックを見つけ、位置と向きを推定する。

1. OWLv2 に front カメラの画像と言葉 ("a wooden block") を与え、ブロックの矩形を検出させる (AI の推論)
2. 矩形の中の深度画像を 3 次元の点に戻し、テーブルより少し高い点だけを残す
3. 点の重心をブロックの位置、点の広がりの主軸 (長辺) をブロックの向きとする

シミュレーション内部のブロックの位置 (正解) は使わない。
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = "google/owlv2-base-patch16-ensemble"
QUERY = "a wooden block"


@dataclass
class Detection:
    xy: np.ndarray  # ブロック中心の位置 [m]
    yaw: float  # 長辺の向き [rad]
    score: float  # 検出の確からしさ (OWLv2 の出力)
    box: np.ndarray  # 画像上の矩形 [x0, y0, x1, y1] (ピクセル)
    n_points: int


class BlockDetector:
    def __init__(self, model: mujoco.MjModel, camera: str = "front", height: int = 480, width: int = 640,
                 block_dims=(0.075, 0.025, 0.015), threshold: float = 0.08, device: str | None = None):
        self.model = model
        self.camera = camera
        self.h, self.w = height, width
        self.block_dims = np.array(block_dims)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self.net = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(self.device).eval()
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        cam = model.camera(camera)
        self.f = 0.5 * height / np.tan(np.radians(cam.fovy[0]) / 2)

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(RGB 画像, 深度画像 [m])。"""
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    @torch.no_grad()
    def detect_boxes(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """OWLv2 で矩形を検出する。(矩形 (N, 4), スコア (N,))。"""
        inputs = self.processor(text=[[QUERY]], images=rgb, return_tensors="pt").to(self.device)
        outputs = self.net(**inputs)
        # OWLv2 は画像を正方形にパディングして処理するので、長い辺の大きさで座標を戻す
        side = max(self.h, self.w)
        res = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=torch.tensor([[side, side]], device=self.device))[0]
        boxes, scores = res["boxes"], res["scores"]
        keep = nms(boxes, scores, iou_threshold=0.3)
        return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()

    def _points_world(self, depth: np.ndarray, data: mujoco.MjData, box: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = np.clip(box, 0, [self.w - 1, self.h - 1, self.w - 1, self.h - 1]).astype(int)
        v, u = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        d = depth[y0:y1 + 1, x0:x1 + 1]
        # MuJoCo のカメラは -z 方向を向き、y が上
        cam = np.stack([(u - self.w / 2 + 0.5) * d / self.f, -(v - self.h / 2 + 0.5) * d / self.f, -d], axis=-1)
        cid = self.model.camera(self.camera).id
        R, p = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]
        return cam.reshape(-1, 3) @ R.T + p

    def detect(self, data: mujoco.MjData) -> list[Detection]:
        """テーブルに平らに置かれたブロックを検出する (積んだブロックは厚さの判定で除かれる)。"""
        rgb, depth = self.render(data)
        self.last_rgb = rgb  # 検出に使った画像 (動画用)
        boxes, scores = self.detect_boxes(rgb)
        self.rejected = []  # 捨てた矩形と理由 (確認用)
        length, width, thick = self.block_dims
        found = []
        for box, score in zip(boxes, scores):
            pts = self._points_world(depth, data, box)
            # 平らに置かれた 1 段目のブロックの上面の点だけを使う (側面の点を含めると重心がカメラ側にずれる)
            top = pts[(pts[:, 2] > thick * 0.85) & (pts[:, 2] < thick * 1.3)]
            if len(top) < 30:
                self.rejected.append(("few points", float(score), len(top)))
                continue
            xy = top[:, :2]
            center = xy.mean(axis=0)
            evals, evecs = np.linalg.eigh(np.cov((xy - center).T))
            major = evecs[:, 1]
            # 点の広がりがブロック 1 個分でなければ、ロボットの一部や複数個をまとめた矩形とみなして捨てる
            extent = 4 * np.sqrt(evals)  # 一様分布なら 辺の長さ ≈ 2√3 σ ≈ 3.5σ
            if not (0.5 * length < extent[1] < 1.5 * length and extent[0] < 1.5 * width * 2):
                self.rejected.append(("extent", float(score), np.round(extent * 100, 1).tolist()))
                continue
            found.append(Detection(center, float(np.arctan2(major[1], major[0])), float(score), box, len(top)))
        # 同じブロックを指す重複を除く (位置が 2cm 以内ならスコアの高い方を残す)
        found.sort(key=lambda d: -d.score)
        unique = []
        for d in found:
            if all(np.linalg.norm(d.xy - u.xy) > 0.02 for u in unique):
                unique.append(d)
        return unique

    def annotate(self, rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """検出結果の矩形を画像に描く (動画用)。"""
        out = rgb.copy()
        for d in detections:
            x0, y0, x1, y1 = d.box.astype(int)
            for t in range(2):
                out[max(y0 - t, 0):y1 + t + 1, [max(x0 - t, 0), min(x1 + t, self.w - 1)]] = (60, 220, 90)
                out[[max(y0 - t, 0), min(y1 + t, self.h - 1)], max(x0 - t, 0):x1 + t + 1] = (60, 220, 90)
        return out

    def close(self) -> None:
        self.renderer.close()
