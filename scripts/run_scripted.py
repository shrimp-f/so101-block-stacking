"""スクリプトポリシーの成功率を測る。

使い方 (コンテナ内):
    python scripts/run_scripted.py --n-blocks 3 --episodes 20 [--video 2]
出力:
    成功率・平均段数・失敗の内訳
    outputs/phase3/scripted_n{N}_ep{K}.mp4 (先頭 --video 個のエピソード, front | wrist)
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import imageio
import numpy as np

from so101_stack.envs.stack_env import StackEnv
from so101_stack.scripted.stack_policy import PlanningError, ScriptedStacker

OUT_DIR = Path("outputs/phase3")


def run_episode(env: StackEnv, seed: int, record: bool, best_effort: bool = False) -> tuple[dict, str, list]:
    env.reset(seed=seed)
    policy = ScriptedStacker(env, best_effort=best_effort)
    policy.reset()
    frames, info, reason = [], {}, "timeout"
    while True:
        try:
            action = policy.act()
        except PlanningError as e:
            reason = f"planning: {e}"
            break
        if action is None:
            reason = "policy finished without success"
            break
        _, _, terminated, truncated, info = env.step(action)
        if record:
            frames.append(np.concatenate([env.render_camera("front"), env.render_camera("wrist")], axis=1))
        if terminated:
            reason = "success"
            break
        if truncated:
            break
    return info, reason, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=int, default=0, help="動画を保存するエピソード数")
    parser.add_argument("--best-effort", action="store_true", help="置く経路が厳密に解けなくても近づいて置く")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = StackEnv(n_blocks=args.n_blocks, obs_type="state")
    reasons = Counter()
    heights = []
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        record = ep < args.video
        info, reason, frames = run_episode(env, args.seed + ep, record, args.best_effort)
        reasons[reason.split(" at ")[0]] += 1
        heights.append(info.get("max_stack_height", 0))
        print(f"ep {ep:3d} seed {args.seed + ep}: {reason}, max height {heights[-1]}, steps {env._step_count}")
        if record:
            path = OUT_DIR / f"scripted_n{args.n_blocks}_ep{ep}.mp4"
            imageio.mimwrite(path, frames, fps=30)
    env.close()

    n_success = reasons["success"]
    print(f"\nsuccess {n_success}/{args.episodes} ({n_success / args.episodes:.0%}), "
          f"mean max height {np.mean(heights):.2f}, {time.perf_counter() - t0:.0f}s")
    for reason, n in reasons.most_common():
        print(f"  {n:3d} x {reason}")


if __name__ == "__main__":
    main()
