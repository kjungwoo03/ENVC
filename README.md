# ENVC

Official Repository of paper "Event-guided Neural Video Compression"

## Setup

The supported environment uses Python 3.10, PyTorch 2.11.0 with CUDA 13.0, and CuPy 14.1.1.

```bash
conda create -n ENVC python=3.10
conda activate ENVC
pip install -r requirements.txt
```

## Evaluation

Run inside a GPU allocation or on a GPU server:

```bash
bash eval/EVAL.sh \
  --dataset /path/to/HEVC-D --layout yuv \
  --p-frame-model-path /path/to/envc_weights.pt \
  --i-frame-model-path /path/to/intra_weights.pt \
  --quality 0 --num-frames 96 --intra-period 32 --resume
```

For the four embedded rate points:

```bash
for quality in 0 1 2 3; do
  bash eval/EVAL.sh \
    --dataset /path/to/HEVC-D --layout yuv \
    --p-frame-model-path /path/to/envc_weights.pt \
    --i-frame-model-path /path/to/intra_weights.pt \
    --quality "$quality" --num-frames 96 --intra-period 32 --resume || break
done
```

## Model Checkpoints

Our checkpoints will be released soon!

## Acknowledgments

The implementation includes components derived from DCVC, DCMVC, and VFPSIE. Existing source copyright and license notices are retained. See [LICENSE](LICENSE) and [licenses/](licenses/).
