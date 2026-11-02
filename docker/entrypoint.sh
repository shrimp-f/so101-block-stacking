#!/usr/bin/env bash
set -e

# START_VNC=1 のとき仮想ディスプレイと noVNC を起動する
# ブラウザで http://localhost:6080/vnc.html を開くと画面を操作できる
if [ "${START_VNC:-0}" = "1" ]; then
    export DISPLAY=:99
    Xvfb :99 -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &
    sleep 1
    x11vnc -display :99 -forever -shared -nopw -quiet >/tmp/x11vnc.log 2>&1 &
    websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
    echo "noVNC: http://localhost:6080/vnc.html"
fi

exec "$@"
