"""AI の認識 (OWLv2) だけで積み上げるポリシーの成功率を測る。

使い方 (コンテナ内): python scripts/run_perception.py --n-blocks 2 --episodes 10 [--device cpu]
"""

import argparse
import time

import numpy as np

from so101_stack.envs.stack_env import StackEnv
from so101_stack.perception.block_detector import BlockDetector
from so101_stack.scripted.perception_stacker import PerceptionStacker
from so101_stack.scripted.stack_policy import PlanningError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="物体検出を動かすデバイス (cuda / cpu)")
    args = parser.parse_args()

    env = StackEnv(n_blocks=args.n_blocks, obs_type="state")
    det = BlockDetector(env.model, block_dims=env.cfg.block_dims, device=args.device)
    n_success = 0
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        policy = PerceptionStacker(env, det)
        policy.reset()
        reason, grips = "policy finished without success", []
        while True:
            try:
                action = policy.act()
            except PlanningError as e:
                reason = f"planning: {e}"
                break
            if action is None:
                break
            _, _, terminated, truncated, info = env.step(action)
            if terminated:
                reason = "success"
                break
            if truncated:
                reason = "timeout"
                break
        n_success += reason == "success"
        print(f"ep {ep} seed {args.seed + ep}: {reason}, height {env.stack_height()}, "
              f"detections {[len(f) for _, f, _ in policy.events]}, gripper now {policy.gripper_value():.1f}")
    det.close()
    env.close()
    print(f"\nsuccess {n_success}/{args.episodes} ({n_success / args.episodes:.0%}), {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
