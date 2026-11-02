"""スクリプトポリシーのデモを LeRobotDataset として保存する。成功したエピソードだけを残す。

使い方 (コンテナ内):
    python scripts/record_demos.py --n-blocks 3 --episodes 100 --repo-id local/so101_jenga_stack3
保存先:
    data/<repo-id の "/" 以降>  (Docker ボリューム so101-data)

データの形式 (LeRobot の SO-101 実機と同じキー・単位):
    observation.state          (6,) 関節角 [deg], グリッパ 0〜100
    observation.images.front   (480, 640, 3) 動画
    observation.images.wrist   (480, 640, 3) 動画
    action                     (6,) 関節の目標角 (state と同じ単位)

評価用の seed (0 から) と重ならないよう、既定では seed 100000 から使う。
"""

import argparse
import shutil
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_stack.envs.stack_env import StackEnv
from so101_stack.robot import CONTROL_HZ, JOINT_NAMES
from so101_stack.scripted.stack_policy import PlanningError, ScriptedStacker

CAMERAS = ("front", "wrist")
IMAGE_SIZE = (480, 640)


def task_text(n_blocks: int) -> str:
    return f"Stack {n_blocks} Jenga blocks into a single tower."


def make_features() -> dict:
    names = [f"{j}.pos" for j in JOINT_NAMES]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (*IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def record_episode(env: StackEnv, dataset: LeRobotDataset, seed: int, task: str) -> tuple[bool, int]:
    obs, _ = env.reset(seed=seed)
    policy = ScriptedStacker(env)
    policy.reset()
    steps = 0
    while True:
        try:
            action = policy.act()
        except PlanningError:
            return False, steps
        if action is None:
            return False, steps
        frame = {"observation.state": obs["agent_pos"], "action": action, "task": task}
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = obs["pixels"][cam]
        dataset.add_frame(frame)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated:
            return True, steps
        if truncated:
            return False, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100, help="保存する成功エピソードの数")
    parser.add_argument("--repo-id", default=None, help="既定: local/so101_jenga_stack{n_blocks}")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true", help="同じ名前のデータセットがあれば消して作り直す")
    args = parser.parse_args()

    repo_id = args.repo_id or f"local/so101_jenga_stack{args.n_blocks}"
    root = args.root / repo_id.split("/")[-1]
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists (use --overwrite to replace it)")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=CONTROL_HZ,
        features=make_features(),
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        image_writer_threads=4,
    )
    env = StackEnv(n_blocks=args.n_blocks, obs_type="pixels_agent_pos", cameras=CAMERAS, image_size=IMAGE_SIZE)
    task = task_text(args.n_blocks)

    saved, seed, t0 = 0, args.seed, time.perf_counter()
    while saved < args.episodes:
        t_ep = time.perf_counter()
        ok, steps = record_episode(env, dataset, seed, task)
        if ok:
            dataset.save_episode()
            saved += 1
        else:
            dataset.clear_episode_buffer()
        elapsed = time.perf_counter() - t0
        print(
            f"seed {seed}: {'saved' if ok else 'discarded'} ({steps} steps, {time.perf_counter() - t_ep:.0f}s) "
            f"| {saved}/{args.episodes} saved, elapsed {elapsed / 60:.1f} min",
            flush=True,
        )
        seed += 1

    dataset.finalize()
    env.close()
    print(f"done: {saved} episodes -> {root}")


if __name__ == "__main__":
    main()
