"""提出用のデモ動画 (3 分以内, 1280x720, 30fps) を作る。

シミュレータでエピソードを実行して front | wrist の映像を撮り、字幕を付けて 1 本にまとめる。
SmolVLA の場面は、先に scripts/run_smolvla.py で撮った映像 (outputs/smolvla) を使う。
字幕には日本語フォントが必要なので、Windows のフォントフォルダをマウントして実行する。

使い方 (ホストの PowerShell):
    docker compose run --rm -v C:\\Windows\\Fonts:/winfonts:ro sim python scripts/make_video.py
出力:
    outputs/video/so101_stack_demo.mp4
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from so101_stack.envs.stack_env import StackEnv
from so101_stack.perception.block_detector import BlockDetector
from so101_stack.scripted.perception_stacker import PerceptionStacker
from so101_stack.scripted.stack_policy import GRIPPER_OPEN, PlanningError, ScriptedStacker

FPS = 30
W, H = 1280, 720
VIDEO_Y = 70  # 映像 (1280x480) を置く位置。上が見出し、下が字幕
BG = (24, 27, 31)
ACCENT = (226, 140, 60)
WHITE = (240, 240, 240)
GRAY = (170, 175, 182)
OUT_PATH = Path("outputs/video/so101_stack_demo.mp4")
IMAGES = Path("doc/images")
TASK_TEXT = "Stack 2 Jenga blocks into a single tower."
FADE_FRAMES = 12  # 場面の切り替えのクロスフェード (0.4 秒)
CAM2_LABELS = ((12, 12, "front カメラ"), (652, 12, "手首カメラ"))
# 方法 C の映像 (top | wrist | side, 1920x480) を 1280x320 に縮めて上下中央に置いたときのラベル
CAM3_LABELS = ((12, 92, "真上カメラ"), (439, 92, "手首カメラ"), (866, 92, "横カメラ"))


def fit_three(frame: np.ndarray) -> np.ndarray:
    """カメラ 3 台を横に並べた 1920x480 の映像を、1280x480 の枠に収める。"""
    small = np.asarray(Image.fromarray(frame).resize((1280, 320), Image.BILINEAR))
    out = np.zeros((480, 1280, 3), np.uint8)
    out[:] = (14, 16, 19)
    out[80:400] = small
    return out


class Fonts:
    def __init__(self, font_dir: Path):
        self.dir = font_dir
        self.cache = {}

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self.cache:
            name = "meiryob.ttc" if bold else "meiryo.ttc"
            self.cache[key] = ImageFont.truetype(str(self.dir / name), size)
        return self.cache[key]


def run_episode(env: StackEnv, seed: int, best_effort: bool = False) -> tuple[list, np.ndarray, list, bool]:
    """1 エピソード実行し、(front | wrist の映像, 行動の列, 各ステップの段数, 成功したか) を返す。"""
    env.reset(seed=seed)
    policy = ScriptedStacker(env, best_effort=best_effort)
    policy.reset()
    frames, actions, heights, success = [], [], [], False
    while True:
        try:
            action = policy.act()
        except PlanningError:
            break
        if action is None:
            break
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(np.concatenate([env.render_camera("front"), env.render_camera("wrist")], axis=1))
        actions.append(action)
        heights.append(env.stack_height())
        if terminated:
            success = True
            break
        if truncated:
            break
    return frames, np.array(actions), heights, success


def record(n_blocks: int, seed: int, best_effort: bool = False, want_success: bool = True,
           max_tries: int = 5) -> tuple[list, np.ndarray, int, list]:
    """成功 (want_success=False なら失敗) するエピソードが出るまで seed を進めて撮る。
    (映像, 行動の列, seed, 各ステップの段数) を返す。"""
    env = StackEnv(n_blocks=n_blocks, obs_type="state")
    try:
        for s in range(seed, seed + max_tries):
            frames, actions, heights, success = run_episode(env, s, best_effort)
            print(f"n_blocks={n_blocks} seed={s} best_effort={best_effort}: "
                  f"{'success' if success else 'fail'}, height {heights[-1]}, {len(frames)} frames")
            if success == want_success:
                return frames, actions, s, heights
    finally:
        env.close()
    raise RuntimeError(f"{n_blocks} 段積みで望む結果が seed {seed}〜{seed + max_tries - 1} で出なかった")


def record_perception(seed: int, max_tries: int = 5) -> tuple[list, list, int]:
    """AI の認識だけで積むポリシーで、成功するエピソードを撮る。(映像, 検出イベント, seed)。"""
    env = StackEnv(n_blocks=2, obs_type="state")
    det = BlockDetector(env.model, block_dims=env.cfg.block_dims)
    try:
        for s in range(seed, seed + max_tries):
            env.reset(seed=s)
            policy = PerceptionStacker(env, det)
            policy.reset()
            frames, success = [], False
            while True:
                try:
                    action = policy.act()
                except PlanningError:
                    break
                if action is None:
                    break
                _, _, terminated, truncated, _ = env.step(action)
                frames.append(np.concatenate([env.render_camera("front"), env.render_camera("wrist")], axis=1))
                if terminated:
                    success = True
                    break
                if truncated:
                    break
            print(f"perception seed={s}: {'success' if success else 'fail'}, {len(frames)} frames")
            if success:
                return frames, [(n, found, det.annotate(rgb, found)) for n, found, rgb in policy.events], s
    finally:
        det.close()
        env.close()
    raise RuntimeError("AI の認識による 2 段積みが成功しなかった")


def load_smolvla_run(out_dir: Path) -> tuple[list, dict, dict]:
    """run_smolvla.py の出力から、最初のエピソードの映像・その結果・全体の集計を読む。"""
    results = json.loads((out_dir / "results.json").read_text())
    stats = {"n": len(results), "success": sum(r["success"] for r in results),
             "chunk_s": float(np.mean([r["chunk_infer_s"] for r in results if r["chunk_infer_s"]]))}
    r = results[0]
    frames = list(imageio.mimread(out_dir / f"ep0_seed{r['seed']}.mp4", memtest=False))
    return frames, r, stats


def first_grasp_step(actions: np.ndarray) -> int:
    """グリッパを閉じ始めたステップ (= 1 個目を掴む瞬間)。"""
    opened = np.flatnonzero(actions[:, 5] >= GRIPPER_OPEN - 1)[0]
    return int(opened + np.flatnonzero(actions[opened:, 5] < GRIPPER_OPEN - 5)[0])


class Composer:
    """字幕付きのフレームを作って動画に書き込む。同じ字幕の下地はキャッシュする。"""

    def __init__(self, writer, fonts: Fonts):
        self.writer = writer
        self.fonts = fonts
        self.cache = {}
        self.n_frames = 0
        self.last = None  # 最後に書いたフレーム
        self.fade_from = None  # 次に書くフレームとの間をクロスフェードでつなぐ元のフレーム

    def transition(self) -> None:
        """場面の区切り。次の場面の最初のフレームを、直前のフレームから溶け込ませて始める。"""
        self.fade_from = self.last

    def _write(self, arr: np.ndarray) -> None:
        if self.fade_from is not None:
            a = self.fade_from.astype(np.float32)
            b = arr.astype(np.float32)
            for k in range(1, FADE_FRAMES + 1):
                t = k / (FADE_FRAMES + 1)
                t = t * t * (3 - 2 * t)  # ゆっくり始まってゆっくり終わる
                self.writer.append_data((a * (1 - t) + b * t).astype(np.uint8))
                self.n_frames += 1
            self.fade_from = None
        self.writer.append_data(arr)
        self.n_frames += 1
        self.last = arr

    def _layout(self, title: str, lines: tuple, speed: str, labels: bool) -> np.ndarray:
        key = (title, lines, speed, labels)
        if key in self.cache:
            return self.cache[key]
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8, VIDEO_Y - 1], fill=ACCENT)
        d.text((24, VIDEO_Y // 2), title, font=self.fonts.get(30, bold=True), fill=WHITE, anchor="lm")
        if speed:
            f = self.fonts.get(22, bold=True)
            x1 = W - 20
            x0 = x1 - d.textlength(speed, font=f) - 28
            d.rounded_rectangle([x0, 16, x1, VIDEO_Y - 16], radius=8, fill=ACCENT)
            d.text(((x0 + x1) / 2, VIDEO_Y // 2), speed, font=f, fill=(20, 20, 20), anchor="mm")
        y = VIDEO_Y + 480 + 22
        for i, line in enumerate(lines):
            f = self.fonts.get(30 if i == 0 else 24, bold=(i == 0))
            d.text((W // 2, y), line, font=f, fill=WHITE if i == 0 else GRAY, anchor="ma")
            y += 48 if i == 0 else 38
        arr = np.asarray(img).copy()
        self.cache[key] = (arr, labels)
        return self.cache[key]

    def _label_layer(self, labels: tuple) -> Image.Image:
        key = ("labels", labels)
        if key not in self.cache:
            layer = Image.new("RGBA", (W, 480), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            f = self.fonts.get(20, bold=True)
            for x, y, text in labels:
                w = d.textlength(text, font=f)
                d.rounded_rectangle([x, y, x + w + 20, y + 32], radius=6, fill=(0, 0, 0, 150))
                d.text((x + 10, y + 16), text, font=f, fill=WHITE, anchor="lm")
            self.cache[key] = layer
        return self.cache[key]

    def add(self, frame: np.ndarray, title: str, lines=(), speed: str = "等倍", labels=CAM2_LABELS,
            repeat: int = 1) -> None:
        """frame は 1280x480。labels は (x, y, 文字) の並び (映像の上に置くカメラ名)。"""
        base, _ = self._layout(title, tuple(lines), speed, bool(labels))
        out = base.copy()
        view = Image.fromarray(frame)
        if labels:
            view = view.convert("RGBA")
            view.alpha_composite(self._label_layer(labels))
            view = view.convert("RGB")
        out[VIDEO_Y:VIDEO_Y + 480] = np.asarray(view)
        for _ in range(repeat):
            self._write(out)

    def card(self, seconds: float, background: np.ndarray | None, title_lines: list, sub_lines: list) -> None:
        """文字だけ (または背景画像の上に文字) の画面。"""
        img = Image.new("RGB", (W, H), BG)
        if background is not None:
            bg = Image.fromarray(background).resize((W, int(background.shape[0] * W / background.shape[1])))
            bg = Image.blend(bg, Image.new("RGB", bg.size, BG), 0.65)
            img.paste(bg, (0, (H - bg.height) // 2))
        d = ImageDraw.Draw(img)
        total = len(title_lines) * 70 + len(sub_lines) * 46 + 20
        y = (H - total) // 2
        for line in title_lines:
            d.text((W // 2, y), line, font=self.fonts.get(48, bold=True), fill=WHITE, anchor="ma")
            y += 70
        y += 20
        for line in sub_lines:
            d.text((W // 2, y), line, font=self.fonts.get(28), fill=ACCENT if line.startswith("再生") else WHITE,
                   anchor="ma")
            y += 46
        self._hold(np.asarray(img), seconds)

    def card_list(self, seconds: float, title: str, sections: list) -> None:
        """見出しと箇条書きの画面。sections は (小見出し, [行, ...]) のリスト。"""
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8, VIDEO_Y - 1], fill=ACCENT)
        d.text((24, VIDEO_Y // 2), title, font=self.fonts.get(30, bold=True), fill=WHITE, anchor="lm")
        y = VIDEO_Y + 40
        for heading, lines in sections:
            d.text((80, y), heading, font=self.fonts.get(30, bold=True), fill=ACCENT)
            y += 54
            for line in lines:
                d.text((110, y), line, font=self.fonts.get(26), fill=WHITE)
                y += 44
            y += 26
        self._hold(np.asarray(img), seconds)

    def _hold(self, arr: np.ndarray, seconds: float) -> None:
        for _ in range(int(seconds * FPS)):
            self._write(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-dir", type=Path, default=Path("/winfonts"))
    parser.add_argument("--seed", type=int, default=0, help="2 段積みの最初のシード")
    parser.add_argument("--vla-dir", type=Path, default=Path("outputs/smolvla_base_matched"),
                        help="方法 C (学習なしの SmolVLA, 3 カメラ・正規化補正あり) の run_smolvla.py の出力")
    parser.add_argument("--vla-raw-dir", type=Path, default=Path("outputs/smolvla_base"),
                        help="正規化を補正する前の run_smolvla.py の出力 (比較用)")
    parser.add_argument("--perception-rate", default="16 / 20", help="方法 B の 2 段積みの成功率 (字幕用)")
    args = parser.parse_args()
    fonts = Fonts(args.font_dir)

    # --- 撮影 ---
    main_frames, main_actions, seed, _ = record(2, args.seed)
    perc_frames, perc_events, perc_seed = record_perception(args.seed + 1)
    vla_frames, vla_result, vla_stats = load_smolvla_run(args.vla_dir)
    raw_frames, _, _ = load_smolvla_run(args.vla_raw_dir)

    # --- 編集 ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    c = Composer(writer, fonts)

    # 1. タイトル
    seq = imageio.imread(IMAGES / "stack2_sequence.png")[..., :3]
    c.card(7, seq, ["LeRobot SO-101 × AI", "ジェンガ 2 段積み:3 つの方法を比較"],
           ["A. 正解の位置:シミュレータ内部の位置を使う(AI なし)",
            "B. AI 物体検出:OWLv2 がカメラ画像からブロックの位置を推定",
            "C. VLA(学習なし):SmolVLA が画像と指示文から動作を推論",
            "再生速度: 画面右上に表示(等倍 / 0.5倍速 / 2倍速)"])

    c.transition()

    # 2. 環境の紹介 (リセット直後の画像)
    scene = np.concatenate([imageio.imread(IMAGES / "scene_front.png")[..., :3],
                            imageio.imread(IMAGES / "scene_wrist.png")[..., :3]], axis=1)
    c.add(scene, "シミュレータ環境(MuJoCo)", ("SO-101 アームとジェンガのブロック(7.5 × 2.5 × 1.5cm)",
                                              "ブロックの位置と向きは毎回ランダム。ロボット正面 25cm に積み上げる"),
          speed="", repeat=FPS * 4)

    c.transition()

    # 3. 方法 A: 正解の位置による 2 段積み
    title = "方法 A:正解の位置(AI なし)"
    lines = ("シミュレータ内部のブロックの位置を使い、ルールベース + 逆運動学で動かす",
             "2 段積みの成功率 100 / 100")
    for f in main_frames:
        c.add(f, title, lines)
    c.add(main_frames[-1], title, ("成功:2 段に積み上がった", lines[1]), repeat=FPS)

    c.transition()

    # 4. 掴む瞬間をスローで (0.5 倍速 = 各フレームを 2 回)
    g = first_grasp_step(main_actions)
    lines = ("工夫:片側の指しか動かないため、掴む位置を開く側へ 8mm ずらす",
             "固定側の指がブロックに乗り上げない/指の内側に当たり判定を追加")
    for f in main_frames[max(0, g - 45):g + 30]:
        c.add(f, "方法 A:掴む瞬間(スロー)", lines, speed="0.5倍速", repeat=2)

    c.transition()

    # 5. 方法 B: AI 物体検出 (OWLv2) で位置を推定して積む
    title = "方法 B:AI 物体検出(OWLv2)で位置を推定"
    lines = ("正解の位置は使わず、AI が画像から見つけた位置・向きで方法 A と同じ手順で積む",
             f"2 段積みの成功率 {args.perception_rate}(seed {perc_seed})")
    events = {n: (found, img) for n, found, img in perc_events}
    for k, f in enumerate(perc_frames):
        if k in events:
            found, img = events[k]
            still = np.concatenate([img, f[:, 640:]], axis=1)
            c.add(still, title, (f"AI が「a wooden block」を検出:{len(found)} 個(緑の枠)",
                                 "枠の中の深度から位置・向きを計算。タワー上のものを除き、近いものから掴む"),
                  speed="一時停止", repeat=FPS * 2)
        c.add(f, title, lines)
    c.add(perc_frames[-1], title, ("成功:AI の認識だけで 2 段に積み上がった", lines[1]), repeat=FPS)

    c.transition()

    # 6. 方法 C: VLA (SmolVLA, 学習なし) の推論
    title = "方法 C:VLA(SmolVLA、学習なし)"
    rate = f"2 段積みの成功率 {vla_stats['success']} / {vla_stats['n']}"
    # 正規化を補正する前: 出力がほぼ 0° になり、腕を前に伸ばして止まる (先頭 8 秒を 2 倍速で)
    for f in raw_frames[:240:2]:
        c.add(f, title + ":補正前", ("公開モデルをそのまま使うと、関節角の正規化が効かず出力がほぼ 0°",
                                     "腕を前に伸ばして止まるだけ → 正規化をデモデータの統計値に差し替えた"),
              speed="2倍速")
    c.transition()
    head = (f"入力:カメラ画像 3 枚 + 関節角 + 指示文「{TASK_TEXT}」",
            f"出力:関節の目標角 50 ステップ分(推論 1 回 約 {vla_stats['chunk_s']:.1f} 秒)。追加学習なし")
    for k, f in enumerate(vla_frames):
        lines = head if k < len(vla_frames) // 2 else (
            "ブロックの近くまで腕を下ろすが、掴まずに関係のない姿勢へ移る", rate + "(このシミュレータとタスクは未学習)")
        c.add(fit_three(f), title, lines, labels=CAM3_LABELS)
    c.add(fit_three(vla_frames[-1]), title, (f"失敗:20 秒でブロックを掴めず(seed {vla_result['seed']})", rate),
          labels=CAM3_LABELS, repeat=FPS * 2)

    c.transition()

    # 7. 3 つの方法の比較
    c.card_list(9, "3 つの方法の比較(2 段積み)", [
        ("成功率", ["A. 正解の位置(AI なし):100 / 100",
                   f"B. AI 物体検出(OWLv2)+ ルールベース:{args.perception_rate}",
                   f"C. VLA(SmolVLA、学習なし):{vla_stats['success']} / {vla_stats['n']}"]),
        ("わかったこと", ["認識だけ AI に任せる B は、追加学習なしでも 80% まで届く",
                         "動作まで AI に任せる C は、このロボット・見た目・タスクに合わせた追加学習が必要"]),
    ])

    c.transition()

    # 8. 課題と今後
    c.card_list(10, "課題と今後", [
        ("方法 C", ["方法 A の動作をお手本として記録し、SmolVLA を追加学習して A・B と比べる"]),
        ("方法 B", ["2 個目の見落としと位置ずれを減らす(手首カメラでも確認して掴み直す)"]),
        ("実機へ", ["色・照明・摩擦などをランダムに変えて学習し、実機の SO-101 でも動くようにする"]),
    ])

    c.transition()

    # 9. まとめ
    c.card(9, None, ["まとめ"],
           [f"A. 正解の位置 100 / 100、B. AI 物体検出 {args.perception_rate}、"
            f"C. VLA(学習なし){vla_stats['success']} / {vla_stats['n']}",
            "認識を AI に任せる B は、追加学習なしでも 80% に届いた",
            "動作まで AI に任せる C には、このタスクに合わせた追加学習が必要",
            "github.com/shrimp-f/so101-block-stacking"])

    writer.close()
    print(f"saved: {OUT_PATH} ({c.n_frames} frames, {c.n_frames / FPS:.1f}s)")


if __name__ == "__main__":
    main()
