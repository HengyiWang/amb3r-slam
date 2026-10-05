<h1 align="center">AMB3R-SLAM: Kilometer-scale SLAM<br>with Hierarchical Backend</h1>


<div align="center">

### [Paper](https://arxiv.org/abs/2609.19518) | [Project page](https://hengyiwang.github.io/projects/amber-slam)

</div>

```bibtex
@article{wang2026amb3rslam,
  title={AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2609.19518},
  year={2026}
}
```



<a id="installation"></a>

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/HengyiWang/amb3r-slam.git
cd amb3r-slam
```

**2. Create a conda environment**

```bash
conda create -n amb3r-slam python=3.9 -y
conda activate amb3r-slam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # match your CUDA
pip install -r requirements.txt
bash thirdparty/build.sh  
```


**3. Download weights**

```bash
bash checkpoints/download.sh       # WITH_SALAD=1 for the SALAD checkpoint
```

This fetches `DA3NESTED-GIANT-LARGE-1.1`, `DA3-SMALL` and `ORBvoc.txt` into `./checkpoints/` (plus `dino_salad.ckpt` with `WITH_SALAD=1`). `--model_name omega` also needs `checkpoints/vggt_omega_1b_512.pt`.

---

## Demo

### Usage

```bash
python run.py --dataset demo --data_path <path-to-image-folder> --save_map
```

### Outputs

Results are written to `outputs/demo_mono_<model>/`:

- **`traj_<folder>.npz`** : the trajectory, one camera-to-world pose per frame
- **`map_<folder>.ply`** : a coloured point cloud of the map, every pixel (with `--save_map`)
- **`map_<folder>_stride.ply`** : the same map at one pixel in `--pixel_stride` along each axis (with `--save_map`)

### Options

- **`--pixel_stride n`** : the pixel stride of the `_stride` cloud (default 4)
- **`--save_every n`** : keep one of every n mapped frames (default 2)
- **`--map_max_points n`** : cap the size of the cloud
- **`--fps r`** : the capture rate, if the frames were captured below 10 Hz, please specify it to avoid submap geometry issues (default 10 Hz)
- **`--stride n`** : keep every n-th frame



---

<a id="benchmark"></a>

## Benchmark

### Usage

We keep the command used for each dataset:

```bash
bash scripts/run_eval.sh [DATASET ...]
```

You can run each individual dataset like:

```bash
# monocular
python run.py --dataset tum   --data_path $TUM
python run.py --dataset kitti --data_path $KITTI

# RGB-D / stereo
python run.py --dataset tum   --data_path $TUM   --sensor rgbd
python run.py --dataset kitti --data_path $KITTI --sensor stereo

# LiDAR
python run.py --dataset kitti --data_path $KITTI --sensor lidar
python run.py --dataset vbr   --data_path $VBR   --sensor lidar
```

### Flags

- **`--sensor rgbd|stereo|lidar`** : the input modality (default monocular)
- **`--model_name omega`** : use VGGT-Ω instead of Depth Anything 3
- **`--calib`** : give the dataset's intrinsics to the system. It runs uncalibrated otherwise

## License

The AMB3R-SLAM pipeline (everything outside `thirdparty/`) is released under the [Apache License 2.0](LICENSE). The third-party packages under `thirdparty/` and the model weights are not covered by it; please follow the license of each one respectively.

## Acknowledgement

Our code builds upon several amazing open-source projects:

- **Models:** [AMB3R](https://github.com/HengyiWang/amb3r), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [VGGT-Ω](http://vggt-omega.github.io/)
- **Loop retrieval & matching:** [DBoW2](https://github.com/dorian3d/DBoW2), [DPV-SLAM](https://github.com/princeton-vl/DPVO), [SALAD](https://github.com/serizba/salad), [MegaLoc](https://github.com/gmberton/MegaLoc), [ALIKED](https://github.com/Shiaoming/ALIKED)
- **LiDAR & evaluation:** [KISS-ICP](https://github.com/PRBonn/kiss-icp), [evo](https://github.com/MichaelGrupp/evo)

We sincerely thank the authors for their open-source contributions!

---

## Citation

If you find our code or paper useful, please consider citing:

```bibtex
@article{wang2026amb3rslam,
  title={AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical Backend},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2609.19518},
  year={2026}
}
```
