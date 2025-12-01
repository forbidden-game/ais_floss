# FLOSS-AIS: Flow Matching-based AIS Signal Collision Separation

A Flow Matching-based system for separating collided AIS signals in satellite receiver scenarios.

## Quick Deployment on Remote GPU Server

### Step 1: Clone the Repository

```bash
git clone <your-repo-url> floss_ais
cd floss_ais
```

### Step 2: Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell
```

### Step 3: Install Dependencies

```bash
uv sync
```

### Step 4: Train Model

Choose config based on your GPU:

```bash
# For 40GB+ GPU (A100, H100)
uv run python scripts/train.py --config-name=large_gpu

# For 24GB GPU (A10, 3090, 4090)
uv run python scripts/train.py --config-name=medium_gpu

# For 16GB GPU (default config)
uv run python scripts/train.py

# For quick test
uv run python scripts/train_simple.py --train_size 1000 --num_epochs 5
```

### Step 5: Evaluate

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --num_samples 100
```

---

## GPU Configurations

| GPU VRAM | Config | encoder_dim | layers | batch_size | Estimated Time (100 epochs) |
|----------|--------|-------------|--------|------------|----------------------------|
| 40GB+ | large_gpu.yaml | 512 | 6 | 32 | ~4-6 hours |
| 24GB | medium_gpu.yaml | 384 | 6 | 16 | ~6-8 hours |
| 16GB | default.yaml | 128 | 4 | 4 | ~10-12 hours |

---

## Project Structure

```
floss_ais/
├── src/
│   ├── data/
│   │   ├── ais_generator.py      # AIS signal generation (GMSK modulation)
│   │   ├── channel_model.py      # Satellite channel model
│   │   └── collision_generator.py # Collision data generation
│   ├── models/
│   │   ├── encoder.py            # IQ encoder
│   │   ├── flow_matching.py      # Flow Matching core
│   │   ├── decoder.py            # IQ decoder
│   │   └── floss_ais.py          # Complete model
│   ├── training/
│   │   └── trainer.py            # Trainer
│   └── utils/
│       └── metrics.py            # SI-SNR/SNR metrics
├── scripts/
│   ├── train.py                  # Hydra training script
│   ├── train_simple.py           # Simple training script
│   └── evaluate.py               # Evaluation script
├── configs/
│   ├── default.yaml              # Default (16GB GPU)
│   ├── medium_gpu.yaml           # 24GB GPU
│   └── large_gpu.yaml            # 40GB+ GPU
└── checkpoints/                  # Model checkpoints
```

---

## Training Strategy

### 1. Flow Matching Core Principle

FLOSS-AIS uses Flow Matching as the core generative model framework, rather than traditional diffusion models. Flow Matching learns a vector field `v(t, x_t, condition)` that gradually transforms a noise distribution into the target signal distribution.

**Training Process:**

1. **Linear Interpolation Path**: Given ground truth source signal `s` and noise `z`, construct intermediate states via linear interpolation:
   ```
   x_t = (1 - t) * z + t * s,  t in [0, 1]
   ```

2. **Target Vector Field**: The ground truth vector field is:
   ```
   v*(t, x_t) = s - z
   ```

3. **Training Objective**: The network learns to predict the vector field by minimizing MSE loss:
   ```
   L_flow = E[||v_theta(t, x_t, condition) - v*||^2]
   ```

### 2. Network Architecture

**Vector Field Network (VectorFieldNetwork)**:
- Input: current state `x_t`, time step `t`, condition (mixture signal encoding)
- Structure: Multi-layer Transformer Blocks
- Each Block contains:
  - Self-Attention: handles temporal dependencies
  - Cross-Attention: fuses mixture signal condition
  - Feed-Forward: non-linear transformation
  - Time Modulation: injects time information into features

**Encoder-Decoder**:
- IQEncoder: encodes IQ signal into feature sequence
- IQDecoder: decodes separated features back to IQ signal

### 3. Loss Function

Total loss consists of three parts:

```
L_total = lambda_flow * L_flow + lambda_recon * L_recon + lambda_consist * L_consist
```

| Loss Term | Weight | Description |
|-----------|--------|-------------|
| `L_flow` | 1.0 | Flow Matching vector field prediction loss |
| `L_recon` | 0.1 | Reconstruction loss (MSE between separated and ground truth signals) |
| `L_consist` | 0.1 | Mixture consistency loss (sum of separated signals should equal mixture) |

**Permutation Invariant Training (PIT)**:
- Solves the source permutation ambiguity problem
- K=2: enumerate both permutations, take the one with smaller loss
- K>2: use Hungarian algorithm for optimal matching

### 4. Sampling/Inference Strategy

During inference, use ODE solvers to generate separated signals from noise:

**Euler Method**:
```python
for step in range(num_steps):
    t = step / num_steps
    v = vector_field(x_t, t, condition)
    x_t = x_t + v * dt
```

**Midpoint Method** (more accurate):
```python
for step in range(num_steps):
    t = step / num_steps
    v1 = vector_field(x_t, t, condition)
    x_mid = x_t + v1 * (dt / 2)
    v2 = vector_field(x_mid, t + dt/2, condition)
    x_t = x_t + v2 * dt
```

**Initialization**: Use mixture signal mean + small noise for faster convergence.

### 5. Optimization Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| Optimizer | AdamW | Adam with weight decay |
| Learning Rate | 1e-4 | Initial learning rate |
| Weight Decay | 0.01 | L2 regularization |
| LR Scheduler | CosineAnnealing | Cosine annealing, minimum lr/100 |
| Gradient Clipping | 1.0 | Prevents gradient explosion |

### 6. Data Generation

**AIS Signal**:
- Bit rate: 9600 bps
- Sample rate: 96000 Hz
- Modulation: GMSK (BT=0.4, h=0.5)
- Frame structure: Preamble + Flag + Data + CRC + Flag

**Satellite Channel**:
- Doppler shift: +/-3800 Hz
- Delay: 0-100 samples
- SNR range: -5 to 20 dB
- Amplitude variation: 0.5 to 1.5

**Collision Configuration**:
- Number of sources: 2 (default)
- Overlap ratio: 30% to 100%
- SIR range: -10 to 10 dB

---

## Advanced Usage

### Override Config with Hydra

```bash
uv run python scripts/train.py \
    training.batch_size=64 \
    training.num_epochs=200 \
    model.encoder_dim=512
```

### Train with wandb Logging

```bash
uv add wandb
uv run python scripts/train.py use_wandb=true project_name=floss-ais
```

### Resume Training

```bash
# Load from checkpoint (manual)
uv run python scripts/train.py  # Will auto-load from checkpoints/latest.pt if exists
```

---

## Evaluation Metrics

- **SI-SNR (Scale-Invariant SNR)**: Scale-invariant signal-to-noise ratio, primary metric
- **SNR**: Signal-to-noise ratio
- **BER (Bit Error Rate)**: Bit error rate (requires enabling bit decoder)

---

## Training Results Reference

Typical loss curve for small-scale test (1000 samples, 5 epochs):

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 0 | 1.21 | 1.07 |
| 1 | 0.92 | 0.86 |
| 2 | 0.80 | 0.82 |
| 3 | 0.75 | 0.81 |
| 4 | 0.73 | 0.77 |

---

## References

- FLOSS: Flow Matching for Source Separation
- AIS Technical Specifications (ITU-R M.1371)
- Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport
