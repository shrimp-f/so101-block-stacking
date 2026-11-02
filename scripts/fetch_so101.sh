#!/usr/bin/env bash
# TheRobotStudio/SO-ARM100 リポジトリから SO-101 の MJCF とメッシュを取得し assets/so101/ に配置する。
# 使い方 (コンテナ内): bash scripts/fetch_so101.sh
set -euo pipefail

REPO=https://github.com/TheRobotStudio/SO-ARM100.git
DEST=assets/so101
TMP=$(mktemp -d)

git clone --depth 1 --filter=blob:none --sparse "$REPO" "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set Simulation/SO101

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$TMP/repo/Simulation/SO101/." "$DEST/"
git -C "$TMP/repo" rev-parse HEAD > "$DEST/SOURCE_COMMIT"
rm -rf "$TMP"

echo "fetched into $DEST (commit $(cat $DEST/SOURCE_COMMIT))"
ls "$DEST"
