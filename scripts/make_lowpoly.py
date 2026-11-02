"""SO-101 の描画用メッシュを間引き、低ポリゴン版の MJCF を作る。

元のメッシュは約 32 万ポリゴンあり、CPU 描画 (llvmpipe) では 5 fps 程度しか出ない。
見た目用 (class="visual") の geom だけを間引いたメッシュに差し替え、衝突判定用の geom は元のメッシュのまま残す。

使い方 (コンテナ内):
    python scripts/make_lowpoly.py [--ratio 0.1]
出力:
    assets/so101/assets/lowpoly/*.stl
    assets/so101/so101_lowpoly.xml
    assets/so101/scene_lowpoly.xml
"""

import argparse
import re

import trimesh

from so101_stack.robot import ASSET_DIR, LOWPOLY_SCENE_XML

SRC_XML = ASSET_DIR / "so101_new_calib.xml"
DST_XML = ASSET_DIR / "so101_lowpoly.xml"
MESH_DIR = ASSET_DIR / "assets"
LOWPOLY_DIR = MESH_DIR / "lowpoly"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.1, help="残すポリゴンの割合")
    args = parser.parse_args()

    xml = SRC_XML.read_text()
    mesh_names = re.findall(r'<mesh file="([^"]+)\.stl"/>', xml)
    LOWPOLY_DIR.mkdir(exist_ok=True)

    total_src = total_dst = 0
    for name in mesh_names:
        mesh = trimesh.load(MESH_DIR / f"{name}.stl")
        mesh.merge_vertices()  # STL は頂点が共有されておらず、そのままでは間引きが効かない
        target = max(200, int(len(mesh.faces) * args.ratio))
        low = mesh.simplify_quadric_decimation(face_count=target) if len(mesh.faces) > target else mesh
        low.export(LOWPOLY_DIR / f"{name}.stl")
        total_src += len(mesh.faces)
        total_dst += len(low.faces)
        print(f"  {name:40s} {len(mesh.faces):7d} -> {len(low.faces):6d} faces")

    # 描画用メッシュのアセットを追加し、class="visual" の geom だけ差し替える
    vis_assets = "".join(f'\n    <mesh name="{n}_vis" file="lowpoly/{n}.stl"/>' for n in mesh_names)
    xml = xml.replace("<asset>", "<asset>" + vis_assets, 1)
    xml = re.sub(r'(<geom type="mesh" class="visual"[^>]*mesh=")([^"]+)(")', r"\1\2_vis\3", xml)
    xml = xml.replace('<mujoco model="so101_new_calib">', '<mujoco model="so101_lowpoly">', 1)
    DST_XML.write_text(xml)

    # 公式の scene.xml (床・照明) を低ポリゴン版のロボットで読み込む版。
    # CPU 描画向けに影マップを小さくし、アンチエイリアスを切る (640x480 で約 4 fps → 18 fps)
    scene = (ASSET_DIR / "scene.xml").read_text().replace("so101_new_calib.xml", DST_XML.name)
    scene = scene.replace("<visual>", '<visual>\n        <quality shadowsize="2048" offsamples="0" />', 1)
    LOWPOLY_SCENE_XML.write_text(scene)

    print(f"total {total_src} -> {total_dst} faces, wrote {DST_XML}, {LOWPOLY_SCENE_XML}")


if __name__ == "__main__":
    main()
