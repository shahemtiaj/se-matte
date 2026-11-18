#!/bin/zsh
# SE Matte – one-time setup on a Mac (Apple Silicon recommended).
#
#   ./engine/setup_mac.sh            full setup (Python env, third-party code, model weights, panel)
#   ./engine/setup_mac.sh --no-models   skip the large model downloads (they also download on first use)
#
# Needs: Homebrew (for ffmpeg and uv), ~6 GB free disk, internet for this one-time setup only.
# After setup everything runs offline.
set -e
cd "${0:A:h}"                 # engine/
ENGINE="$PWD"
ROOT="${ENGINE:h}"
MODELS=1
[[ "$1" == "--no-models" ]] && MODELS=0

say() { print -P "%F{cyan}==>%f $1"; }

# ------------------------------------------------------------------ tools
say "Checking tools"
command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v uv >/dev/null || brew install uv
command -v git >/dev/null || { echo "git is required (xcode-select --install)"; exit 1; }

# ------------------------------------------------------------------ Python 3.11 environment
say "Creating the Python 3.11 environment (engine/.venv)"
uv python install 3.11
[[ -d .venv ]] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# ------------------------------------------------------------------ third-party model code (pinned)
say "Fetching third-party model code (engine/vendor)"
mkdir -p vendor
fetch() {   # name url commit
  if [[ ! -d "vendor/$1/.git" ]]; then
    rm -rf "vendor/$1"
    git clone --quiet --filter=blob:none "$2" "vendor/$1"
  fi
  git -C "vendor/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "vendor/$1" checkout --quiet "$3"
  echo "   $1 @ $3"
}
fetch sam2               https://github.com/facebookresearch/sam2.git        2b90b9f5ceec907a1c18123530e92e794ad901a4
fetch MatAnyone          https://github.com/pq-yang/MatAnyone.git             e5ddc534c1fff9bb9e54cf476095d29071b7cb4f
fetch RobustVideoMatting https://github.com/PeterL1n/RobustVideoMatting.git   53d74c6826735f01f4406b5ca9075eee27bec094
fetch BiRefNet           https://github.com/ZhengPeng7/BiRefNet.git           f7079632d681ecf12161fcea8905682934e59e51

# ------------------------------------------------------------------ model weights
if (( MODELS )); then
  say "Downloading model weights (one time, ~2.5 GB)"
  mkdir -p models/rvm
  for v in resnet50 mobilenetv3; do
    [[ -f models/rvm/rvm_$v.pth ]] || curl -fL --progress-bar -o models/rvm/rvm_$v.pth \
      "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_$v.pth"
  done
  SEMATTE_OFFLINE=0 HF_HOME="$ENGINE/models/hf" .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt")
for repo in ("PeiqingYang/MatAnyone", "hustvl/vitmatte-small-composition-1k"):
    snapshot_download(repo)
print("   models ready")
PY
fi

# ------------------------------------------------------------------ native Fast-mode helper
say "Building the native Fast-mode helper (engine/bin/mac/semfast)"
if command -v swiftc >/dev/null; then
  ./native/build_mac.sh
elif [[ -x bin/mac/semfast ]]; then
  echo "   swiftc not found – using the prebuilt helper from the release"
else
  echo "   swiftc not found: install the Xcode Command Line Tools (xcode-select --install) for Fast mode"
fi

# ------------------------------------------------------------------ After Effects panel
say "Installing the After Effects panel"
EXT="$HOME/Library/Application Support/Adobe/CEP/extensions"
mkdir -p "$EXT"
ln -sfn "$ROOT/panel" "$EXT/com.shahemtiaj.sematte"
for v in 9 10 11 12; do defaults write com.adobe.CSXS.$v PlayerDebugMode 1; done   # allow the unsigned panel

say "Done. Restart After Effects, then open Window → Extensions → SE Matte."
