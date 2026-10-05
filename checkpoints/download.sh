#!/usr/bin/env bash
# Fetches the checkpoints (and ALIKED's weights) into place. The DA3 weights also download
# on first use (from the Hugging Face hub) if skipped here; the rest are needed only as noted.
set -euo pipefail
cd "$(dirname "$0")"

# Depth Anything 3: back-end (6.8 GB) and front-end.
for m in DA3NESTED-GIANT-LARGE-1.1 DA3-SMALL; do
    huggingface-cli download "depth-anything/$m" --local-dir "$m"
done

# ORB vocabulary, for DBoW2 loop retrieval (the default).
if [ ! -f ORBvoc.txt ]; then
    wget -q https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
    tar -xzf ORBvoc.txt.tar.gz && rm ORBvoc.txt.tar.gz
fi

# ALIKED keypoints, for the covisibility gate.
ALIKED=../thirdparty/ALIKED/models/aliked-n16.pth
if [ ! -f "$ALIKED" ]; then
    mkdir -p "$(dirname "$ALIKED")"
    wget -q -O "$ALIKED" https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth
fi

# Optional: SALAD retrieval (retrieval.method=salad).
if [ "${WITH_SALAD:-0}" = 1 ] && [ ! -f dino_salad.ckpt ]; then
    wget -q https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
fi

# Optional: VGGT-Omega (--model_name omega) expects vggt_omega_1b_512.pt here.
[ -f vggt_omega_1b_512.pt ] || echo 'note: vggt_omega_1b_512.pt not present (needed only for --model_name omega)'
