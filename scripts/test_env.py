"""Phase 2: 環境の動作確認。

1. リセット直後の front / wrist カメラ画像を保存する
2. ランダムな行動で長時間動かし、シミュレーションが破綻しないか確認する
   (NaN、MuJoCo の警告、ブロックの貫通・飛び出し)

使い方 (コンテナ内):
    python scripts/test_env.py [--steps 1000 --video-steps 300]
出力:
    outputs/phase2/reset_front.png, reset_wrist.png
    outputs/phase2/random_actions.mp4
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import mujoco
import numpy as np

import so101_stack.envs  # noqa: F401  環境を登録する

OUT_DIR = Path("outputs/phase2")


def check_physics(env, max_penetration: list[float]) -> str | None:
    d = env.unwrapped.data
    if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
        return "NaN/Inf in state"
    for i, w in enumerate(d.warning):
        if w.number > 0:
            return f"MuJoCo warning: {mujoco.mjtWarning(i).name}"
    pos = env.unwrapped.block_poses()[:, :3]
    if np.any(pos[:, 2] < -0.005):
        return f"block fell through the table: z={pos[:, 2].min():.4f}"
    if np.any(np.abs(pos[:, :2]) > 1.0) or np.any(pos[:, 2] > 1.0):
        return "block flew away"
    if d.ncon:
        max_penetration.append(-min(c.dist for c in d.contact[: d.ncon]))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--video-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make("so101_stack/StackBlocks-v0", obs_type="pixels_agent_pos", max_episode_steps=10**6)
    obs, info = env.reset(seed=args.seed)
    for cam, img in obs["pixels"].items():
        imageio.imwrite(OUT_DIR / f"reset_{cam}.png", img)
    print(f"reset: agent_pos={np.round(obs['agent_pos'], 1)}, info={info}")
    env.close()

    # ランダム行動: 一様乱数の目標角を 0.5 秒ごとに切り替える (毎周期だと振動するだけなので)
    env = gym.make("so101_stack/StackBlocks-v0", obs_type="state", max_episode_steps=10**6)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    penetration: list[float] = []
    frames = []
    error = None
    for t in range(args.steps):
        if t % 15 == 0:
            action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        if t < args.video_steps:
            frames.append(env.unwrapped.render())
        error = check_physics(env, penetration)
        if error:
            print(f"step {t}: {error}")
            break
    imageio.mimwrite(OUT_DIR / "random_actions.mp4", frames, fps=30)
    print(f"random actions: {t + 1} steps, error={error}")
    print(f"  max penetration {max(penetration, default=0) * 1000:.2f} mm, final info={info}")
    env.close()


if __name__ == "__main__":
    main()
