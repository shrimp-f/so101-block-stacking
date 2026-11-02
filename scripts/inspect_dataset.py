"""保存した LeRobotDataset を読み込んで中身を確認する。

使い方 (コンテナ内):
    python scripts/inspect_dataset.py --repo-id local/so101_jenga_stack3
出力:
    エピソード数・フレーム数・特徴量の形
    outputs/phase3/<データセット名>_sample.png  (エピソード 0 の先頭・中間・末尾の front | wrist 画像)
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT_DIR = Path("outputs/phase3")


def to_image(t) -> np.ndarray:
    """(C, H, W) の float [0, 1] テンソルを (H, W, C) の uint8 画像にする。"""
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    name = args.repo_id.split("/")[-1]

    ds = LeRobotDataset(args.repo_id, root=args.root / name)
    print(f"episodes={ds.num_episodes}, frames={ds.num_frames}, fps={ds.fps}")
    item = ds[0]
    for key, value in item.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else value
        print(f"  {key}: {shape}")

    ep = ds.meta.episodes[0]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    rows = []
    for idx in (start, (start + end) // 2, end - 1):
        frame = ds[idx]
        rows.append(np.concatenate([to_image(frame["observation.images.front"]),
                                    to_image(frame["observation.images.wrist"])], axis=1)[::2, ::2])
        print(f"  frame {idx}: state={np.round(frame['observation.state'].numpy(), 1)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_sample.png"
    imageio.imwrite(path, np.concatenate(rows, axis=0))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
