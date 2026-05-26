# Fast-FoundationStereo — 저장소 참고 문서

> 에이전트/개발자용 내부 레퍼런스. CVPR 2026, NVIDIA NVlabs 공식 구현.  
> 논문: [Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](https://arxiv.org/abs/2512.11130)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | **실시간** 제로샷 스테레오 시차(disparity) 추정 → 깊이/포인트클라우드 |
| 선행 | [FoundationStereo](https://github.com/NVlabs/FoundationStereo) 대비 ~10× 빠름, 정확도 유사 |
| 파라미터 | ~14.6M (`model_card.md`) |
| 입력 | 정렬(rectified)된 좌/우 RGB 스테레오 이미지 (0–255, PyTorch 경로) |
| 출력 | 시차 맵 (1/1 해상도 upsample), 선택적으로 depth / PLY |
| 라이선스 | NVIDIA Source Code License (`LICENSE.txt`) |

**가속 전략 (논문 요약)**  
1. Knowledge distillation — 하이브리드 백본 → EdgeNeXt 단일 student  
2. Blockwise NAS — cost filtering 설계  
3. Structured pruning — iterative refinement(GRU) 축소  

---

## 2. 디렉터리 구조

```
Fast-FoundationStereo/
├── core/                    # 모델 핵심 구현
│   ├── foundation_stereo.py # FastFoundationStereo, hourglass, TRT 래퍼
│   ├── extractor.py         # Feature(EdgeNeXt), ContextNetSharedBackbone
│   ├── submodule.py         # Conv/Attention, GWC·concat volume, Triton GWC
│   ├── update.py            # SelectiveConvGRU, disparity refinement
│   ├── geometry.py          # Combined_Geo_Encoding_Volume
│   ├── distill_block.py     # NAS/pruning용 ForwardHelper (체크포인트 복원)
│   └── utils/
│       ├── utils.py         # InputPadder, bilinear_sampler(1d)
│       └── frame_utils.py   # Middlebury/KITTI 등 데이터 I/O (학습용 유틸)
├── scripts/                 # 데모, ONNX/TRT, 프로파일링
├── Utils.py                 # 루트 공용: AMP, depth2xyz, vis_disparity, Open3D
├── demo_data/               # left.png, right.png, K.txt (intrinsic+baseline)
├── docker/                  # CUDA 12.4 + TRT 개발 환경
├── weights/                 # 체크포인트 (gitignore, Google Drive에서 다운로드)
├── docs/                    # 본 문서
├── readme.md                # 사용자 README (대문자 README.md와 동일 내용)
├── model_card.md            # NVIDIA 모델 카드
└── requirements.txt
```

**Git에서 무시되는 것** (`.gitignore`): `weights/*`, `*.pth`, `cfg.yaml`, `output/`, `__pycache__`

---

## 3. 추론 파이프라인 (PyTorch)

### 3.1 데이터 흐름

```
left/right RGB (B,3,H,W) 0–255
    → normalize_image()  # ImageNet mean/std on /255
    → Feature (EdgeNeXt, 좌우 concat 배치)
        → features_left[0..3]  @ 1/4, 1/8, 1/16, 1/32
    → GWC volume + concat volume → corr_stem → cost_agg (hourglass)
        → classifier → softmax → init_disp @ 1/4
    → ContextNetSharedBackbone → net_list, inp_list (+ SAM/CAM)
    → Combined_Geo_Encoding_Volume (geo_feat)
    → valid_iters × BasicSelectiveMultiUpdateBlock (GRU + delta_disp)
    → upsample_disp (context_upsample + stem_2x) → full-res disparity
```

### 3.2 핵심 클래스 (`core/foundation_stereo.py`)

| 클래스 | 역할 |
|--------|------|
| `FastFoundationStereo` | 메인 모델. `forward(image1, image2, iters, test_mode, optimize_build_volume)` |
| `FoundationStereoLite` | `FastFoundationStereo` 별칭 |
| `hourglass` | 3D cost volume U-Net + `CostVolumeDisparityAttention` |
| `TrtFeatureRunner` | TRT 1단계: feature + stem_2 |
| `TrtPostRunner` | TRT 2단계: concat volume 이후 ~ disparity |
| `TrtRunner` | 두 `.engine` 로드 후 Triton GWC + post 실행 |
| `FoundationStereo` | 빈 placeholder (`pass`) — 체크포인트 호환용 |

**체크포인트 로딩 호환** (22–26행): `sys.modules`에 `foundation_stereo_ori.*` → `core.*` 매핑.  
`distill_block.py`는 `from foundation_stereo_ori.submodule import FeatureAtt` 사용.

**`forward` 주요 인자**

- `iters`: refinement 반복 수 (CLI `--valid_iters`)
- `test_mode=True`: 마지막 iteration disparity만 반환
- `optimize_build_volume`: `'pytorch1'` (데모) \| `'triton'` (프로파일링)
- `init_disp`: hierarchical 모드에서 coarse 초기값
- `low_memory`: geo volume 샘플링 시 `bilinear_sampler1d` 사용

**`run_hierachical`**: 저해상도 추론 → upscale → pad 보정 → full-res `forward` (대형 이미지용, `--hiera 1`)

### 3.3 서브모듈 요약

| 파일 | 핵심 |
|------|------|
| `extractor.py` | `timm edgenext_small` + FPN deconv. `d_out` 4스케일 채널. `vit_size`로 vit_feat_dim 결정 |
| `submodule.py` | `build_gwc_volume_*`, `build_concat_volume_*`, `disparity_regression`, `context_upsample`, Attention 블록 |
| `update.py` | `BasicSelectiveMultiUpdateBlock`: motion encoder + `SelectiveConvGRU` + `DispHead` |
| `geometry.py` | `Combined_Geo_Encoding_Volume`: geo volume + all-pairs corr 피라미드, disparity 기준 샘플링 |
| `distill_block.py` | `ForwardHelper` / `PostForwardHelper` — NAS로 치환된 hourglass 서브그래프 복원 |

### 3.4 정규화 (중요)

**PyTorch 경로** (`normalize_image` in `foundation_stereo.py`):

```python
mean = [0.485, 0.456, 0.406]  # 입력은 0–255 RGB
std  = [0.229, 0.224, 0.225]
# (img/255 - mean) / std
```

**단일 ONNX/TRT 경로** (`make_single_onnx.py`): 모델 **내부** 정규화 제거. 외부에서 0–255 픽셀 기준 ImageNet:

```
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]
normalized = (pixel - mean) / std
```

---

## 4. 체크포인트 & 설정

### 4.1 가중치

- Google Drive: README `weights/` 링크
- 예: `weights/23-36-37/model_best_bp2_serialize.pth`
- 형식: `torch.load(..., weights_only=False)` → **전체 `nn.Module` 직렬화** (state_dict만 아님)
- 같은 폴더에 **`cfg.yaml`** 필수 (gitignore, 다운로드 번들에 포함)

### 4.2 `cfg.yaml` (추정 키 — 실제 파일은 가중치 폴더 참고)

스크립트에서 읽고 CLI로 덮어씀:

| 키 | 용도 |
|----|------|
| `valid_iters` | GRU refinement 횟수 |
| `max_disp` | cost volume 최대 시차 (기본 192) |
| `hidden_dims` | GRU/context 채널 |
| `n_gru_layers` | GRU 레이어 수 |
| `corr_levels`, `corr_radius` | geo encoding 피라미드 |
| `mixed_precision` | autocast |
| `vit_size` | `vits` / `vitb` / `vitl` (feature dim) |
| `cv_group`, `volume_dim`, `normalize` | GWC/concat 설정 |
| `low_memory` | 메모리 절약 샘플링 |

### 4.3 체크포인트별 속도 (README, 3090, 640×480)

| Checkpoint | valid_iters | PyTorch ms | TRT ms | Peak MB |
|------------|-------------|------------|--------|---------|
| 23-36-37 | 8 / 4 | 49.4 / 41.1 | 23.4 / 18.4 | 653 |
| 20-26-39 | 8 / 4 | 43.6 / 37.5 | 19.4 / 16.4 | 651 |
| 20-30-48 | 8 / 4 | 38.4 / 29.3 | 16.6 / 14.0 | 646 |

---

## 5. 스크립트 맵

| 스크립트 | 용도 |
|----------|------|
| `scripts/run_demo.py` | **메인 PyTorch 데모**: disparity 시각화, depth.npy, cloud.ply |
| `scripts/run_demo_tensorrt.py` | 2단계 TRT (`feature_runner` + `post_runner` + Triton GWC) |
| `scripts/run_demo_single_trt.py` | 단일 ONNX 또는 `.engine` (ORT/TRT) |
| `scripts/make_onnx.py` | `feature_runner.onnx` + `post_runner.onnx` export |
| `scripts/make_single_onnx.py` | 단일 ONNX (GWC를 ONNX 호환 op로) |
| `scripts/profile_speed.py` | PyTorch latency (`optimize_build_volume=triton`) |
| `scripts/profile_speed_tensorrt.py` | TRT latency |
| `scripts/profile_memory.py` | GPU 메모리 피크 |

### 5.1 `run_demo.py` 플로우

1. `cfg.yaml` + argparse → `OmegaConf`
2. `torch.load(model_dir)` → `.cuda().eval()`
3. 이미지 로드, `--scale` 리사이즈, `InputPadder(divis_by=32)`
4. `forward(..., test_mode=True, optimize_build_volume='pytorch1')` 또는 `run_hierachical`
5. `vis_disparity` → `disp_vis.png`
6. `--get_pc 1`: `K.txt` → `depth = fx * baseline / disp` → Open3D PLY

**주요 CLI 플래그**: `--valid_iters`, `--max_disp`, `--scale`, `--hiera`, `--remove_invisible`, `--denoise_cloud`, `--zfar`

### 5.2 ONNX/TRT 두 경로

| 방식 | Export | GWC | 추론 스크립트 |
|------|--------|-----|----------------|
| **2단계** | `make_onnx.py` | Triton (`build_gwc_volume_triton`) — ONNX 밖 | `run_demo_tensorrt.py` |
| **단일** | `make_single_onnx.py` | ONNX 내부 pad/slice/stack | `run_demo_single_trt.py` |

- 입력 H,W는 **32의 배수**
- TRT: `trtexec --fp16` (+ 2단계는 `--useCudaGraph` 권장)
- Export 시 `TORCH_COMPILE_DISABLE=1`, `TORCHDYNAMO_DISABLE=1`

---

## 6. `Utils.py` (루트)

| 심볼 | 설명 |
|------|------|
| `AMP_DTYPE` | `torch.float16` |
| `set_seed` | 재현성 (cudnn deterministic) |
| `depth2xyzmap` | depth + K → (H,W,3) xyz |
| `vis_disparity` | 컬러맵 시각화 (invalid 마스크 지원) |
| `toOpen3dCloud` | Open3D PointCloud (optional `o3d`) |

---

## 7. `demo_data/K.txt` 형식

```
<9 floats: 3x3 intrinsic flattened>
<baseline in meters>
```

예: fx=fy≈754.67, cx≈489.38, cy≈265.16, baseline=0.063

---

## 8. 의존성

### 8.1 `requirements.txt` (pip)

`timm`, `einops`, `omegaconf`, `scipy`, `numpy`, `scikit-image`, `opencv-contrib-python`, `imageio`, `pyyaml`, `open3d`

### 8.2 README 권장 (CUDA)

```
torch==2.6.0, torchvision==0.21.0, xformers (cu124)
```

### 8.3 Docker (`docker/dockerfile`)

- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- conda `python=3.12`, TRT 10.11, `onnxruntime-gpu`, `tensorrt-cu12`, `nvidia-modelopt`
- **Linux/TRT 전용** — Windows 로컬은 PyTorch 데모 위주

### 8.4 선택 (주석 처리됨)

`onnx`, `onnxruntime-gpu`, `tensorrt-cu12`, `nvidia-modelopt[torch]`

### 8.5 런타임 optional

- **Triton**: `build_gwc_volume_triton` — 없으면 import 실패 가능, PyTorch 경로는 `pytorch1` 사용
- **Open3D**: 없으면 PCL 저장/시각화 스킵

---

## 9. 입력 제약 & 운영 팁

- 좌/우 **정렬·왜곡 보정** 필수; 좌측 카메라가 왼쪽 (스왑 금지)
- PNG 권장; 너비 ~1000px 이하 권장 (`--scale 0.5`로 축소 가능)
- 첫 실행은 torch compile 등으로 **느림** → 워밍업 후 벤치마크
- `InputPadder`: 높이/너비를 `divis_by`(기본 32) 배수로 replicate pad
- 오프라인 최고 정확도: [FoundationStereo](https://github.com/NVlabs/FoundationStereo)

---

## 10. 코드 수정 시 자주 건드리는 위치

| 작업 | 파일 |
|------|------|
| 아키텍처/forward 변경 | `core/foundation_stereo.py` |
| 백본/특징 추출 | `core/extractor.py` |
| cost volume / hourglass | `core/submodule.py`, `hourglass` in foundation_stereo |
| refinement 반복/GRU | `core/update.py`, `valid_iters` |
| ONNX 호환 volume | `scripts/make_single_onnx.py` 내 `_build_*_onnx` |
| TRT 분할 | `TrtFeatureRunner`, `TrtPostRunner`, `make_onnx.py` |
| 데모 I/O·후처리 | `scripts/run_demo.py`, `Utils.py` |
| 패딩/샘플링 | `core/utils/utils.py` |

---

## 11. 외부 데이터셋 & 인용

- Pseudo-label: [Stereo4D](https://huggingface.co/datasets/nvidia/ffs_stereo4d)
- 평가: Middlebury, ETH3D, KITTI (README 참고)

```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```

---

## 12. 빠른 명령 참조

```bash
# PyTorch 데모
python scripts/run_demo.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output/ --valid_iters 8 --max_disp 192 --get_pc 1

# 단일 ONNX export
python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output/ --height 480 --width 640 --valid_iters 8 --max_disp 192
```

---

*마지막 정리: 저장소 파일 기준 (2026-05). 가중치 폴더의 `cfg.yaml`은 로컬 다운로드 후 확인.*
