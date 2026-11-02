"""AI によるブロック認識 (OWLv2 + 深度) の精度を、シミュレーションの正解と比べて確かめる。

使い方 (コンテナ内): python scripts/test_detector.py [--seeds 10]
出力: 位置・向きの誤差、outputs/perception/detect_seed{N}.png (検出した矩形を描いた画像)
"""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from so101_stack.envs.stack_env import StackEnv
from so101_stack.perception.block_detector import BlockDetector
from so101_stack.scripted.stack_policy import block_yaw

OUT_DIR = Path("outputs/perception")


def yaw_error(a: float, b: float) -> float:
    """長辺の向きの差 (180° 回すと同じ形なので 0〜90° に収める)。"""
    d = (a - b) % np.pi
    return np.degrees(min(d, np.pi - d))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-blocks", type=int, default=2)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = StackEnv(n_blocks=args.n_blocks, obs_type="state")
    det = BlockDetector(env.model, block_dims=env.cfg.block_dims)
    pos_errs, yaw_errs, misses, extras = [], [], 0, 0
    for seed in range(args.seeds):
        env.reset(seed=seed)
        t0 = time.perf_counter()
        found = det.detect(env.data)
        dt = time.perf_counter() - t0
        truth = env.block_poses()
        used = set()
        for pose in truth:
            dists = [np.linalg.norm(d.xy - pose[:2]) for d in found]
            if not dists or min(dists) > 0.03:
                misses += 1
                continue
            k = int(np.argmin(dists))
            used.add(k)
            pos_errs.append(dists[k])
            yaw_errs.append(yaw_error(found[k].yaw, block_yaw(pose[3:])))
        extras += len(found) - len(used)
        print(f"seed {seed}: found {len(found)} / {len(truth)}, {dt:.2f}s, "
              f"scores {[round(d.score, 2) for d in found]}, rejected {det.rejected}")
        if seed < 3:
            rgb, _ = det.render(env.data)
            imageio.imwrite(OUT_DIR / f"detect_seed{seed}.png", det.annotate(rgb, found))
    det.close()
    env.close()
    print(f"\npos err mean {np.mean(pos_errs) * 1000:.1f}mm max {np.max(pos_errs) * 1000:.1f}mm, "
          f"yaw err mean {np.mean(yaw_errs):.1f}deg max {np.max(yaw_errs):.1f}deg, "
          f"missed {misses}, extra {extras}")


if __name__ == "__main__":
    main()
