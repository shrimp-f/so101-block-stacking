"""追加学習した SmolVLA (VLA) をシミュレータで動かし、成功率を測る。

カメラ 2 台の画像・関節角・言葉の指示 ("Stack 2 Jenga blocks into a single tower.") を入力に、
SmolVLA が関節の目標角を 50 ステップ分まとめて出力する (推論)。正解のブロック位置は使わない。

使い方 (コンテナ内):
    python scripts/run_smolvla.py --episodes 10 [--checkpoint outputs/train/smolvla_stack2/checkpoints/last/pretrained_model]
出力:
    成功率、outputs/smolvla/ep{K}.mp4 (先頭 --video 個, front | wrist)
"""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from so101_stack.envs.stack_env import StackEnv

TASK = "Stack 2 Jenga blocks into a single tower."
OUT_DIR = Path("outputs/smolvla")
DATASET = ("local/so101_jenga_stack2", Path("data/so101_jenga_stack2"))


def to_batch(obs: dict, device: str) -> dict:
    batch = {"observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0), "task": [TASK]}
    for cam, img in obs["pixels"].items():
        batch[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/train/smolvla_stack2/checkpoints/last/pretrained_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600, help="1 エピソードの上限 (30Hz で 20 秒)")
    parser.add_argument("--video", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cameras", default="front,wrist",
                        help="camera1, camera2, ... に順に割り当てるシミュレータのカメラ")
    parser.add_argument("--dataset-stats", action="store_true",
                        help="関節角の正規化に、モデル同梱の統計値ではなくデモデータセットの平均・標準偏差を使う")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cameras = tuple(args.cameras.split(","))
    rename = {f"observation.images.{c}": f"observation.images.camera{k}" for k, c in enumerate(cameras, start=1)}

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": rename}})
    if args.dataset_stats:
        # smolvla_base 同梱の統計値はロボットごとの名前 (so100.buffer.action など) で保存されていて、
        # "observation.state" / "action" に当たらず正規化が素通りになる。デモデータセットの統計値に差し替える
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.processor.normalize_processor import to_tensor
        stats = LeRobotDatasetMetadata(DATASET[0], root=DATASET[1]).stats
        stats = {k: stats[k] for k in ("observation.state", "action")}
        for step in [*pre.steps, *post.steps]:
            if hasattr(step, "_tensor_stats"):
                step.stats = stats
                step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)
    for step in [*pre.steps, *post.steps]:
        if hasattr(step, "_tensor_stats"):
            print(type(step).__name__, "stats:", sorted(step._tensor_stats))

    env = StackEnv(n_blocks=2, obs_type="pixels_agent_pos", cameras=cameras, max_episode_steps=args.max_steps)
    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        policy.reset()
        frames, infer_times, success, max_h = [], [], False, 0
        for _ in range(args.max_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                action = post(policy.select_action(pre(to_batch(obs, device))))
            infer_times.append(time.perf_counter() - t0)
            obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            max_h = max(max_h, info.get("max_stack_height", 0))
            if ep < args.video:
                frames.append(np.concatenate([obs["pixels"][c] for c in cameras], axis=1))
            if terminated:
                success = True
                break
            if truncated:
                break
        # 推論 (50 ステップ分の行動をまとめて出す回) の時間。それ以外のステップはキューから取り出すだけ
        slow = [t for t in infer_times if t > 0.05]
        r = {"seed": seed, "success": success, "max_height": int(max_h), "steps": len(infer_times),
             "chunk_infer_s": float(np.mean(slow)) if slow else None}
        results.append(r)
        print(r, flush=True)
        if frames:
            imageio.mimwrite(out_dir / f"ep{ep}_seed{seed}.mp4", frames, fps=30)
    env.close()
    n = sum(r["success"] for r in results)
    print(f"\nsuccess {n}/{len(results)}, mean max height {np.mean([r['max_height'] for r in results]):.2f}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
