#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON310="${PYTHON310:-python3.10}"
"$PYTHON310" -c 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 required"'
"$PYTHON310" configure.py
mkdir -p repos/isaac45_probe
if [[ ! -d repos/isaac45_probe/.venv ]]; then
    "$PYTHON310" -m venv repos/isaac45_probe/.venv
fi
PY=./repos/isaac45_probe/.venv/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --extra-index-url https://pypi.nvidia.com \
    isaacsim==4.5.0.0 isaacsim-app==4.5.0.0 isaacsim-core==4.5.0.0 \
    isaacsim-gui==4.5.0.0 isaacsim-replicator==4.5.0.0 \
    isaacsim-extscache-kit==4.5.0.0 isaacsim-extscache-physics==4.5.0.0
if [[ ! -d repos/IsaacLab ]]; then
    git clone --branch v2.1.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git repos/IsaacLab
fi
test "$(git -C repos/IsaacLab rev-parse HEAD)" = 90b79bb2d44feb8d833f260f2bf37da3487180ba || {
    echo 'Isaac Lab version differs from the tested revision; not overwriting your checkout.' >&2; exit 1;
}
"$PY" -m pip install -e repos/IsaacLab/source/isaaclab -e repos/IsaacLab/source/isaaclab_assets \
    -e repos/IsaacLab/source/isaaclab_tasks -e repos/IsaacLab/source/isaaclab_rl
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r requirements.txt
"$PY" -m pip check
echo 'Environment installed. Run collect.py doctor; a new-machine short capture is still required.'
