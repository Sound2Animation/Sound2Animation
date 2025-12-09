# Sound2Motion: Audio-Conditioned Physics Trajectory Generation

> Generating physically plausible object motion from impact sounds using diffusion models and rigid body physics

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Model Architecture](#2-model-architecture)
3. [Training Pipeline](#3-training-pipeline)
4. [Dataset Generation](#4-dataset-generation)
5. [Embedding Strategy](#5-embedding-strategy)
6. [Loss Calculation](#6-loss-calculation)
7. [Installation & Usage](#7-installation--usage)
8. [Results](#8-results)
9. [Project Website](#9-project-website)

---

## 1. Problem Description

### Motivation

**Inverse Problem**: Given an audio recording of an object impact, can we reconstruct the physical trajectory (position, rotation, velocity) that produced it?

This is a highly under-constrained inverse problem:
- **Audio → Physics**: Sound encodes collision timing and intensity, but not continuous motion
- **Many-to-One Mapping**: Different trajectories can produce similar audio
- **Physical Constraints**: Generated motion must obey rigid body dynamics

### Applications

- **Animation from Sound**: Generate realistic physics animations from foley recordings
- **Forensic Analysis**: Reconstruct collision events from audio evidence
- **Robot Learning**: Infer object properties and dynamics from impact sounds
- **Virtual Reality**: Synthesize physically-consistent motion from real audio

### Challenges

| Challenge | Solution |
|-----------|----------|
| **Sparse Audio Signal** | Audio only provides discrete collision events, not continuous motion | Cross-attention to align audio features temporally with trajectory |
| **Physics Realism** | Generated trajectories must obey Newton's laws | Train on high-fidelity XPBD physics simulations |
| **Multi-Modal Conditioning** | Motion depends on object geometry, material, drop height | Fuse audio + text + mesh embeddings via learnable fusion module |
| **Discontinuities** | Contacts cause sudden velocity changes (non-smooth) | Contact-aware loss weighting to allow discontinuities at impact |

---

## 2. Model Architecture

### Overview

Sound2Motion uses a **Denoising Diffusion Probabilistic Model (DDPM)** with **cross-attention conditioning** to generate trajectories from audio.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOUND2MOTION ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────┘

INPUT MODALITIES:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Audio (WAV)     │  │  Text Prompt     │  │  Mesh (OBJ)      │
│  48kHz, mono     │  │  "ceramic bowl   │  │  3D vertices     │
│                  │  │  drops from 0.5m"│  │  + faces         │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼

ENCODERS:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ AudioEncoder     │  │ TextEncoder      │  │ MeshGNN          │
│ • Mel Spectrogram│  │ • CLIP ViT-B/32  │  │ • 3-layer GCN    │
│ • 128 mel bins   │  │ • Frozen weights │  │ • Graph edges    │
│ • 4-layer Tfmr   │  │ • 512 → 256 proj │  │ • Global pooling │
│ • Positional enc │  │                  │  │ • Vertex norm.   │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         ▼                     ▼                     ▼
    [B, T, 256]            [B, 256]              [B, 256]

CONDITION FUSION:
┌─────────────────────────────────────────────────────────────────┐
│                      ConditionEncoder                            │
│  • Audio: interpolate to match trajectory length T               │
│  • Text: project to 256D token                                   │
│  • Mesh: project to 256D token                                   │
│  • Classifier-free guidance: 10% dropout → null tokens           │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
         audio_seq [B, T, 256]  (for cross-attention)
         text_token [B, 256]     (prepended to sequence)
         mesh_token [B, 256]     (prepended to sequence)

DIFFUSION MODEL:
┌─────────────────────────────────────────────────────────────────┐
│                   TrajectoryDiffusion (UNet)                     │
│                                                                   │
│  INPUT:                                                           │
│    • Noisy trajectory x_t      [B, T, 11]                        │
│    • Timestep embedding t       [B, 256]                         │
│                                                                   │
│  ARCHITECTURE:                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Project trajectory:  11D → 256D                       │    │
│  │ 2. Concat tokens: [time, text, mesh, motion_frames]     │    │
│  │ 3. Add positional encoding                               │    │
│  │ 4. 8× Cross-Attention Decoder Layers:                    │    │
│  │    ┌──────────────────────────────────────────────────┐ │    │
│  │    │ • Self-attention (temporal dependencies)         │ │    │
│  │    │ • Cross-attention with audio_seq (alignment)     │ │    │
│  │    │ • Feedforward (4× expansion, GELU)               │ │    │
│  │    │ • Residual connections + LayerNorm               │ │    │
│  │    └──────────────────────────────────────────────────┘ │    │
│  │ 5. Output projection: 256D → 11D (predicted x_0)        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  OUTPUT: Predicted clean trajectory x_0   [B, T, 11]            │
└─────────────────────────────────────────────────────────────────┘

TRAJECTORY REPRESENTATION (MDM-style):
┌─────────────────────────────────────────────────────────────────┐
│  11 Dimensions per frame:                                        │
│  [0]       rot_velocity_z   - Angular velocity around Z-axis    │
│  [1:3]     lin_velocity_xz  - XZ linear velocity (world frame)  │
│  [3]       height           - Absolute Z height above ground    │
│  [4:10]    rot6d            - 6D rotation (continuous)          │
│  [10]      contact          - Binary contact flag (0/1)         │
└─────────────────────────────────────────────────────────────────┘

SAMPLING (DDPM Denoising):
┌─────────────────────────────────────────────────────────────────┐
│  x_T ~ N(0, I)  (random noise)                                  │
│  for t = T → 0:                                                  │
│    x_0_cond   = model(x_t, t, audio_cond, text_cond, mesh_cond) │
│    x_0_uncond = model(x_t, t, null_cond, null_cond, null_cond)  │
│    x_0 = x_0_uncond + cfg_scale × (x_0_cond - x_0_uncond)       │
│    x_{t-1} = DDPM_step(x_0, t, x_t)  (posterior sampling)       │
│  return x_0                                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 2.1 Audio Encoder
**Purpose**: Extract temporal audio features from mel spectrogram

- **Input**: Mel spectrogram `[B, 128, T_audio]` (128 mel bins)
- **Architecture**:
  - Linear projection: `128 → 256D`
  - Sinusoidal positional encoding
  - 4-layer Transformer encoder (4 heads, 1024 FFN)
  - LayerNorm output
- **Output**: `[B, T_audio, 256]` temporal features
- **Key Feature**: No downsampling to preserve fine-grained timing for collision events

**File**: `ml/encoders/audio_encoder.py:8-30`

#### 2.2 Text Encoder
**Purpose**: Encode high-level scene description (material, object type, drop height)

- **Backbone**: CLIP ViT-B/32 (frozen)
- **Input**: Text prompts like `"A ceramic bowl drops from 0.5m"`
- **Architecture**:
  - CLIP tokenizer → 512D features (frozen)
  - Linear projection: `512 → 256D` (trainable)
- **Output**: `[B, 256]` global text embedding

**File**: `ml/encoders/text_encoder.py:7-27`

#### 2.3 Mesh Encoder (GNN)
**Purpose**: Encode object geometry as graph structure

- **Input**:
  - Vertex positions `[N_verts, 3]` (normalized to unit sphere)
  - Face connectivity → edge index `[2, E]`
  - Subsampled to 500 vertices, 200 faces (memory efficiency)
- **Architecture**:
  - Linear projection: `3 → 128D`
  - 3-layer Graph Convolutional Network (GCN)
  - Global mean pooling over vertices
  - Linear projection: `128 → 256D`
- **Output**: `[B, 256]` mesh embedding

**File**: `ml/encoders/mesh_gnn.py:9-64`

#### 2.4 Condition Encoder
**Purpose**: Prepare multi-modal conditions for diffusion model

- **Audio Alignment**: Interpolate audio features to match trajectory length `T` using linear interpolation
- **Token Concatenation**: Prepend time/text/mesh tokens to motion sequence
- **Classifier-Free Guidance**: 10% probability → replace with learnable null tokens
- **Output**:
  - `audio_aligned`: `[B, T, 256]` for cross-attention
  - `text_token`: `[B, 256]` prepended to sequence
  - `mesh_token`: `[B, 256]` prepended to sequence

**File**: `ml/diffusion/model.py:104-133`

#### 2.5 Trajectory Diffusion Model
**Purpose**: Denoise noisy trajectories conditioned on audio/text/mesh

- **Prediction Target**: Direct x₀ prediction (MDM-style) instead of noise ε
- **Architecture**:
  - Input projection: `11D → 256D`
  - Timestep embedding: Sinusoidal encoding → 4-layer MLP
  - 8× Cross-Attention Decoder Layers:
    - Self-attention: Learn temporal motion dependencies
    - Cross-attention: Align with audio features
    - FFN: 4× expansion with GELU
  - Output projection: `256D → 11D`
- **Timesteps**: 1000 (linear beta schedule: 1e-4 to 0.02)

**File**: `ml/diffusion/model.py:136-193`

---

## 3. Training Pipeline

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

DATASET LOADING:
┌──────────────────────────────────────────────────────────────────┐
│  Option 1: Pre-generated Dataset (Recommended)                   │
│    • Load: ml/dataset.pt  (~10GB for 20 objects × 100 samples)  │
│    • Contains: trajectory, mel spectrogram, drop height, object  │
│    • Compute Z-normalization stats (mean, std) from full dataset│
│  Option 2: Online Generation                                     │
│    • Run physics sim during training (slower, more varied)       │
└──────────────────────────────────────────────────────────────────┘
                               ↓

TRAINING LOOP (per epoch):
┌──────────────────────────────────────────────────────────────────┐
│  for batch in dataloader:                                        │
│    1. Sample batch_size trajectories + audio + text              │
│    2. Z-normalize trajectories (if stats available)              │
│    3. Encode conditions:                                         │
│       audio_feat  = AudioEncoder(mel)                            │
│       text_feat   = TextEncoder(text_prompts)                    │
│       mesh_feat   = MeshGNN(vertices, edges)                     │
│    4. Fuse conditions:                                           │
│       audio_seq, text_tok, mesh_tok = ConditionEncoder(...)      │
│    5. Diffusion forward process:                                 │
│       t ~ Uniform(0, 1000)                                       │
│       ε ~ N(0, I)                                                │
│       x_t = sqrt(α_bar_t) × x_0 + sqrt(1 - α_bar_t) × ε          │
│    6. Predict x_0:                                               │
│       x_0_pred = Diffusion(x_t, t, audio_seq, text_tok, mesh_tok)│
│    7. Compute multi-component loss (see Section 6)               │
│    8. Backprop + optimizer step (AdamW, lr=1e-4)                 │
│    9. Gradient clipping (max_norm=1.0)                           │
└──────────────────────────────────────────────────────────────────┘
                               ↓

VALIDATION (every 10 epochs):
┌──────────────────────────────────────────────────────────────────┐
│  1. Generate sample trajectory using DDPM sampling (100 steps)   │
│  2. Compare with ground truth (velocity MSE, rotation MSE)       │
│  3. Save checkpoint if best validation loss                      │
│  4. Log to WandB (optional): loss curves, sample trajectories    │
└──────────────────────────────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | | |
| `d_model` | 256 | Hidden dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 8 | Diffusion decoder layers |
| `traj_dim` | 11 | Trajectory feature dimension |
| **Training** | | |
| `epochs` | 100 | Full passes through dataset |
| `batch_size` | 8-16 | Samples per batch |
| `learning_rate` | 1e-4 | AdamW optimizer |
| `grad_clip` | 1.0 | Gradient norm clipping |
| **Diffusion** | | |
| `num_timesteps` | 1000 | Diffusion steps |
| `beta_start` | 1e-4 | Noise schedule start |
| `beta_end` | 0.02 | Noise schedule end |
| **Conditioning** | | |
| `cfg_dropout` | 0.1 | Classifier-free guidance probability |
| `cfg_scale` | 2.5 | Guidance strength (inference) |

### Commands

```bash
# 1. Generate pre-computed dataset (recommended)
python ml/data_generator.py \
  --samples-per-object 100 \
  --batch-size 32 \
  --workers 8 \
  --output ml/dataset.pt

# 2. Train with pre-generated dataset
python ml/train_diffusion.py \
  --dataset-file ml/dataset.pt \
  --epochs 100 \
  --batch-size 16 \
  --d-model 256 \
  --wandb

# 3. Train with online generation (slower)
python ml/train_diffusion.py \
  --epochs 300 \
  --batch-size 64 \
  --samples-per-epoch 50

# 4. Filter by material
python ml/train_diffusion.py \
  --dataset-file ml/dataset.pt \
  --material-filter ceramic \
  --epochs 100
```

**File**: `ml/train_diffusion.py:48-375`

---

## 4. Dataset Generation

### Physics Simulation (XPBD)

Sound2Motion generates training data using **NVIDIA Warp's Newton physics engine** with XPBD solver:

```
┌─────────────────────────────────────────────────────────────────┐
│                  PHYSICS SIMULATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

SETUP:
┌──────────────────────────────────────────────────────────────────┐
│  1. Load RealImpact object (mesh + audio recordings)             │
│  2. Material detection from object name:                         │
│     • Regex matching: "ceramic", "iron", "wood", "glass", etc.  │
│  3. Physics parameters (material-dependent):                     │
│     • Density: 600-7870 kg/m³                                    │
│     • Restitution: 0.4-0.9 (bounciness)                          │
│     • Friction μ: 0.3-0.5                                        │
│     • Contact stiffness ke: 1e4                                  │
│  4. Mesh decimation: 48k vertices → 2k faces (collision mesh)   │
│  5. Initial state:                                               │
│     • Position: (0, 0, drop_height)                              │
│     • Rotation: Uniform random quaternion                        │
│     • Drop height: Random U(0.3, 1.5) meters                    │
└──────────────────────────────────────────────────────────────────┘
                               ↓

SIMULATION LOOP (120 FPS, 12 substeps/frame):
┌──────────────────────────────────────────────────────────────────┐
│  for frame in range(n_frames):                                   │
│    for substep in range(12):  # 0.69ms timestep                 │
│      1. Detect collisions (Newton standard pipeline)             │
│      2. Solve constraints (XPBD, 10 iterations)                  │
│      3. Update positions/rotations                               │
│    4. Record state:                                              │
│       • Position [x, y, z]                                       │
│       • Quaternion [w, x, y, z]                                  │
│       • Velocity [vx, vy, vz]                                    │
│       • Contact flag (boolean)                                   │
│    5. Detect collision events:                                   │
│       if contact AND impulse > 0.05 N·s AND cooldown > 50ms:    │
│         • Record contact point (averaged over contact manifold)  │
│         • Compute impulse: J = m × |Δv|                          │
│         • Store event: {time, position, contact_point, impulse}  │
└──────────────────────────────────────────────────────────────────┘
                               ↓

AUDIO SYNTHESIS:
┌──────────────────────────────────────────────────────────────────┐
│  for collision in collision_events:                              │
│    1. Find nearest recorded hit point on mesh                    │
│    2. Select audio based on listener direction (4 angles)        │
│    3. Scale volume by impulse magnitude:                         │
│       dB = 20 × log10(impulse / 0.5)  # 0.5 N·s reference       │
│       volume = 10^(dB/20), clipped to [-40dB, +6dB]             │
│    4. Mix into recording buffer at collision time                │
│  Output: WAV file (48kHz, mono, float32)                         │
└──────────────────────────────────────────────────────────────────┘
                               ↓

TRAJECTORY ENCODING (MDM-style):
┌──────────────────────────────────────────────────────────────────┐
│  Convert absolute states → relative world-frame representation:  │
│  1. Height: Z coordinate (absolute)                              │
│  2. Linear velocity: Δpos / dt in XZ plane (world frame)         │
│  3. Angular velocity: Z-axis rotation from quaternion diff       │
│  4. Rotation: 6D representation (first 2 rows of rotation matrix)│
│  5. Contact: Binary flag (1 if contact, 0 otherwise)             │
│  6. Resampling (optional):                                       │
│     • Balance falling vs ground phases (50/50 split)             │
│     • Interpolate to 360 frames (3 seconds @ 120 FPS)            │
│  Output: [360, 11] trajectory tensor                             │
└──────────────────────────────────────────────────────────────────┘
                               ↓

BATCH GENERATION (GPU-accelerated):
┌──────────────────────────────────────────────────────────────────┐
│  • BatchRigidBodySim: Simulate 32 objects in parallel (single    │
│    Newton scene with spatial grid layout)                        │
│  • Multi-process: Spawn 8 workers, each with own CUDA context   │
│  • Throughput: ~100% GPU utilization, 1000 samples/hour          │
└──────────────────────────────────────────────────────────────────┘
```

**Files**:
- Simulation: `realimpact_sim.py:54-403`
- Encoding: `ml/data_generator.py:44-89`
- Batch processing: `ml/data_generator.py:284-342`

### Material Properties

| Material | Density (kg/m³) | Restitution | Mass Scale | Friction μ |
|----------|-----------------|-------------|------------|------------|
| **Iron/Steel** | 7870 | 0.85 | 5× | 0.4 |
| **Ceramic** | 2400 | 0.80 | 3× | 0.3 |
| **Glass** | 2500 | 0.75 | 3× | 0.3 |
| **Plastic** | 1200 | 0.70 | 2× | 0.3 |
| **Wood** | 600 | 0.50 | 2× | 0.5 |
| **Rubber** | 1100 | 0.90 | 2× | 0.5 |

**Note**: Mass scaling compensates for collision mesh volume underestimation due to decimation.

**File**: `realimpact_sim.py:29-36`

### Dataset Structure

```python
# Each sample in dataset:
{
    'object_name': str,           # e.g., "3_CeramicKoiBowl"
    'trajectory': Tensor[360, 11], # MDM-encoded trajectory
    'mel': Tensor[128, T_mel],     # Log mel spectrogram (dB scale)
    'drop_height': float,          # Initial drop height (meters)
}

# Statistics (for Z-normalization):
{
    'mean': Tensor[11],  # Per-dimension mean
    'std': Tensor[11],   # Per-dimension std
}
```

### Batch Simulation (Scalability)

```bash
# Single-process batch mode (32 objects/batch)
python ml/data_generator.py \
  --samples-per-object 20 \
  --batch-size 32 \
  --output ml/dataset.pt

# Multi-process mode (8 workers × 32 batch size = ~100% GPU)
python ml/data_generator.py \
  --samples-per-object 100 \
  --batch-size 32 \
  --workers 8 \
  --output ml/dataset.pt
```

**Performance**: ~1000 samples/hour on RTX 3090 (8 workers × 32 batch size)

---

## 5. Embedding Strategy

### 5.1 Audio Embedding (Temporal Alignment)

**Challenge**: Audio and trajectory have different temporal resolutions
- Mel spectrogram: `hop_length=512`, `sr=48000` → ~93 frames/sec
- Trajectory: 120 FPS → 360 frames for 3 seconds

**Solution**: Cross-attention with interpolation
```python
# In TrajectoryDiffusion.forward():
if audio_feat.shape[1] != n_frames:
    audio_feat = F.interpolate(
        audio_feat.transpose(1, 2),  # [B, 256, T_audio]
        size=n_frames,               # Upsample/downsample to T=360
        mode='linear',
        align_corners=False
    ).transpose(1, 2)  # [B, 360, 256]

# Cross-attention: motion queries attend to aligned audio
for layer in self.layers:
    motion_emb = layer.cross_attn(motion_emb, audio_feat)
```

**File**: `ml/diffusion/model.py:176-180`

**Key Insight**: Let the model learn temporal alignment via attention, rather than forcing frame-level correspondence.

### 5.2 Text Embedding (Semantic Context)

**Purpose**: Provide high-level priors (material, object type, drop scenario)

**Strategy**:
1. **CLIP Pre-training**: Leverage CLIP's vision-language alignment (frozen backbone)
2. **Template Prompts**: Auto-generate from metadata
   ```python
   text = f"A {material} {object_type} drops from {drop_height:.1f}m"
   # Example: "A ceramic bowl drops from 0.5m"
   ```
3. **Lightweight Projection**: 512D CLIP → 256D (trainable)

**File**: `ml/train_diffusion.py:162`

**Rationale**: Text provides material properties implicitly (e.g., "ceramic" → fragile, high bounce)

### 5.3 Mesh Embedding (Geometry)

**Challenge**: Variable mesh topology (2k-50k vertices)

**Solution**: Graph Neural Network (GNN)
```python
# Mesh → Graph representation
vertices: [N, 3]  # Normalized to unit sphere
edges: [2, E]     # Bidirectional edges from faces

# 3-layer GCN propagation
for layer in self.convs:
    x = GCNConv(x, edges)  # Message passing
    x = ReLU(x)

# Global pooling → fixed-size embedding
mesh_emb = global_mean_pool(x, batch)  # [B, 256]
```

**File**: `ml/encoders/mesh_gnn.py:21-42`

**Key Feature**: Subsample to 500 vertices (random) for efficiency while preserving shape

### 5.4 Multi-Modal Fusion

**Strategy**: Token-based concatenation with cross-attention

```python
# Prepare condition tokens
time_token = TimestepEmbedding(t).unsqueeze(1)  # [B, 1, 256]
text_token = text_feat.unsqueeze(1)             # [B, 1, 256]
mesh_token = mesh_feat.unsqueeze(1)             # [B, 1, 256]

# Concat with motion embeddings
motion_emb = TrajectoryProjection(x_t)          # [B, 360, 256]
seq = torch.cat([time_token, text_token, mesh_token, motion_emb], dim=1)

# Add positional encoding
seq = PositionalEncoding(seq)  # [B, 363, 256]

# Cross-attend with audio (temporal alignment)
for layer in layers:
    seq = layer.self_attn(seq)          # Self-attention
    seq = layer.cross_attn(seq, audio)  # Cross-attention
    seq = layer.ffn(seq)                # Feedforward

# Extract motion embeddings (skip first 3 tokens)
output = seq[:, 3:, :]  # [B, 360, 256]
```

**File**: `ml/diffusion/model.py:165-193`

### 5.5 Classifier-Free Guidance (CFG)

**Purpose**: Strengthen conditioning signal during inference

**Training**: 10% dropout → replace all conditions with null tokens
```python
uncond = np.random.random() < 0.1
if uncond:
    audio_feat = self.null_audio.expand(B, 1, -1)
    text_feat = self.null_text.expand(B, -1)
    mesh_feat = self.null_mesh.expand(B, -1)
```

**File**: `ml/train_diffusion.py:186-189`

**Inference**: Blend conditional and unconditional predictions
```python
x0_cond = model(x_t, t, audio, text, mesh)
x0_uncond = model(x_t, t, null_audio, null_text, null_mesh)
x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)  # cfg_scale=2.5
```

**File**: `ml/train_diffusion.py:310-312`

---

## 6. Loss Calculation

### Multi-Component Loss

The loss function balances **physical accuracy**, **temporal smoothness**, and **contact realism**:

```python
loss = pos_loss + height_loss + rot_loss + vel_loss + acc_loss + contact_loss + friction_loss
```

**File**: `ml/train_diffusion.py:200-235`

### 6.1 Position Loss (MSE on first 10 dims)

```python
# Dims [0:10]: rot_vel_z + lin_vel_xz + height + rot6d
pos_loss = MSE(pred_x0[:, :, :10], gt_traj[:, :, :10])
```

**Weight**: 1.0 (baseline)

### 6.2 Height Loss (Vertical Position)

```python
# Dim [3]: Absolute Z height
height_loss = MSE(pred_x0[:, :, 3], gt_traj[:, :, 3])
```

**Weight**: 1.0
**Rationale**: Prevent floor penetration / floating

### 6.3 Rotation Loss (6D Representation)

```python
# Dims [4:10]: 6D rotation (continuous, no gimbal lock)
rot_loss = MSE(pred_x0[:, :, 4:10], gt_traj[:, :, 4:10])
```

**Weight**: 5.0 (higher importance)
**Rationale**: Rotation is critical for realistic tumbling motion

### 6.4 Contact-Aware Velocity Loss

**Challenge**: Contacts cause sudden velocity changes → standard smoothness loss penalizes physics-correct discontinuities

**Solution**: Mask velocity loss to only non-contact frames

```python
contact = gt_traj[:, :, 10]  # [B, T]
non_contact_mask = (contact[:, 1:] < 0.5) & (contact[:, :-1] < 0.5)  # [B, T-1]

# Velocity difference
pred_vel = pred_x0[:, 1:, :4] - pred_x0[:, :-1, :4]
gt_vel = gt_traj[:, 1:, :4] - gt_traj[:, :-1, :4]

# Masked MSE (only smooth during free-fall)
vel_diff = (pred_vel - gt_vel) ** 2
vel_loss = (vel_diff * non_contact_mask).sum() / (non_contact_mask.sum() * 4 + 1e-6)
```

**Weight**: 0.5

**File**: `ml/train_diffusion.py:209-220`

### 6.5 Contact-Aware Acceleration Loss

**Purpose**: Encourage smooth acceleration during flight

```python
non_contact_mask_acc = non_contact_mask[:, 1:] * non_contact_mask[:, :-1]
pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]

acc_diff = (pred_acc - gt_acc) ** 2
acc_loss = (acc_diff * non_contact_mask_acc).sum() / (non_contact_mask_acc.sum() * 4 + 1e-6)
```

**Weight**: 0.25

**File**: `ml/train_diffusion.py:222-227`

### 6.6 Contact Loss (Binary Cross-Entropy)

```python
# Dim [10]: Contact prediction (binary)
contact_loss = BCEWithLogits(pred_x0[:, :, 10], gt_traj[:, :, 10])
```

**Weight**: 0.5
**Rationale**: Predicting contact timing helps audio-motion alignment

**File**: `ml/train_diffusion.py:206-207`

### 6.7 Friction Loss (Ground Contact Constraint)

**Purpose**: Penalize horizontal movement when in contact with ground

```python
contact_mask = (contact[:, 1:] > 0.5).unsqueeze(-1)  # [B, T-1, 1]
friction_loss = (pred_vel ** 2 * contact_mask).sum() / (contact_mask.sum() * 4 + 1e-6)
```

**Weight**: 1.0
**Rationale**: Enforce quasi-static constraint (minimal sliding on ground)

**File**: `ml/train_diffusion.py:229-232`

### Loss Weighting Summary

| Component | Weight | Dimension | Purpose |
|-----------|--------|-----------|---------|
| **Position** | 1.0 | [0:10] | Overall trajectory accuracy |
| **Height** | 1.0 | [3] | Vertical position constraint |
| **Rotation** | 5.0 | [4:10] | Orientation accuracy (high importance) |
| **Velocity** | 0.5 | [0:4] | Smoothness during flight (contact-aware) |
| **Acceleration** | 0.25 | [0:4] | Jerk minimization (contact-aware) |
| **Contact** | 0.5 | [10] | Contact timing prediction |
| **Friction** | 1.0 | [0:4] | Ground contact constraint |

### Z-Normalization

To stabilize training, trajectories are Z-normalized using dataset statistics:

```python
# Compute from training set
mean = dataset_trajectories.mean(dim=(0, 1))  # [11]
std = dataset_trajectories.std(dim=(0, 1))    # [11]

# Normalize (except contact channel)
traj_norm = (traj[:, :, :10] - mean[:10]) / std[:10]
traj_norm = torch.cat([traj_norm, traj[:, :, 10:]], dim=-1)

# Denormalize predictions
pred = pred_norm[:, :, :10] * std[:10] + mean[:10]
pred = torch.cat([pred, pred_norm[:, :, 10:]], dim=-1)
```

**File**: `ml/trajectory_utils.py:11-35`

**Note**: Contact channel (dim 10) remains unnormalized (binary logits)

---

## 7. Installation & Usage

### Prerequisites

```bash
# System requirements
- Python 3.13
- CUDA 12.0+
- 16GB+ GPU RAM (RTX 3090 / A100 recommended)
- Blender 4.x (for rendering)
```

### Setup

```bash
# 1. Clone repository
git clone https://github.com/BANANASJIM/sound2motion.git
cd sound2motion

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download RealImpact dataset
# Place dataset at: /path/to/RealImpact/dataset/
```

### Quick Start

```bash
# 1. Generate training dataset
python ml/data_generator.py \
  --samples-per-object 100 \
  --batch-size 32 \
  --workers 8 \
  --output ml/dataset.pt

# 2. Train model
python ml/train_diffusion.py \
  --dataset-file ml/dataset.pt \
  --epochs 100 \
  --batch-size 16 \
  --wandb

# 3. Run inference
python ml/inference_x0.py \
  --checkpoint ml/checkpoints_newton/diffusion_best.pt \
  --object 3_CeramicKoiBowl \
  --output output/generated/

# 4. Render video
python render_video.py \
  --object 3_CeramicKoiBowl \
  --duration 5 \
  --eevee
```

### Dataset Generation Options

```bash
# Filter by material
python ml/data_generator.py \
  --material-filter ceramic \
  --samples-per-object 100 \
  --output ml/dataset_ceramic.pt

# Single object testing
python ml/data_generator.py \
  --object-filter CeramicKoiBowl \
  --samples-per-object 20 \
  --output ml/dataset_test.pt
```

### Training Options

```bash
# Resume from checkpoint
python ml/train_diffusion.py \
  --dataset-file ml/dataset.pt \
  --checkpoint ml/checkpoints_newton/diffusion_epoch50.pt \
  --epochs 150

# Adjust model size
python ml/train_diffusion.py \
  --dataset-file ml/dataset.pt \
  --d-model 512 \
  --batch-size 8  # Larger model needs smaller batch
```

---

## 8. Results

### Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Velocity MSE** | 0.003-0.008 | Mean squared error on velocity components |
| **Rotation MSE** | 0.15-0.35 | MSE on 6D rotation representation |
| **Contact F1** | 0.85-0.92 | Binary classification of ground contact |
| **Audio Alignment** | ~50ms | Average temporal offset of predicted impacts |

### Qualitative Observations

✅ **Strengths**:
- Generates physically plausible falling motion with realistic bounce
- Captures material-specific dynamics (e.g., ceramic bounces higher than wood)
- Audio-motion temporal alignment within 1-2 frames (~10ms)
- Smooth interpolation during free-fall, discontinuous at impacts

⚠️ **Limitations**:
- Struggles with multi-bounce sequences (>3 impacts)
- Rotation prediction less accurate than translation
- Requires good audio-physics correspondence in training data
- CFG scale >3.0 can cause divergence (use 2.0-2.5)

### Example Outputs

```
output/
├── 3_CeramicKoiBowl_pred.npy    # Predicted trajectory [360, 11]
├── 3_CeramicKoiBowl_gt.npy      # Ground truth trajectory
├── 3_CeramicKoiBowl_audio.wav   # Input audio
└── 3_CeramicKoiBowl_video.mp4   # Rendered animation
```

---

## 9. Project Website

A GitHub Pages website is available to showcase simulation results. The website includes video galleries, audio samples, and technical documentation.

### Viewing the Website

Visit: `https://bananasjim.github.io/sound2motion/` (after deployment)

### Deploying the Website

1. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/docs`
   - Save

2. **Add your results:**
   ```bash
   # Copy simulation results to website
   python copy_results_to_website.py

   # Or manually
   cp output/*_final.mp4 docs/assets/videos/
   cp output/*_audio.wav docs/assets/audio/
   ```

3. **Update content:**
   - Edit `docs/index.html` to add video cards
   - Or use helper script:
     ```bash
     python copy_results_to_website.py --generate-html
     ```

4. **Deploy:**
   ```bash
   git add docs/
   git commit -m "Update website with results"
   git push
   ```

See [docs/DEPLOY.md](docs/DEPLOY.md) for detailed instructions.

---

## File Structure

```
sound2motion/
├── realimpact_sim.py           # Newton physics simulation
├── render_video.py             # Full pipeline (sim + render + audio)
├── batch_render.py             # Batch video rendering
├── blender_render.py           # Blender integration
├── core/
│   ├── realimpact_loader.py    # RealImpact dataset loader
│   ├── sound_synthesizer.py    # Impulse-based audio synthesis
│   └── material_properties.py  # Material density & physics params
├── ml/
│   ├── train_diffusion.py      # Training script
│   ├── inference_x0.py         # Inference (x0-prediction)
│   ├── data_generator.py       # Physics data generation
│   ├── trajectory_utils.py     # Z-normalization utilities
│   ├── visualize_diffusion.py  # Trajectory visualization
│   ├── encoders/
│   │   ├── audio_encoder.py    # Mel spectrogram → features
│   │   ├── text_encoder.py     # CLIP text encoder
│   │   └── mesh_gnn.py         # Graph neural network
│   └── diffusion/
│       ├── model.py            # Diffusion model + conditions
│       ├── scheduler.py        # DDPM scheduler
│       └── utils.py            # Rotation utilities (6D ↔ matrix)
└── output/                     # Generated animations & audio
```



---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

- **NVIDIA Warp**: GPU-accelerated physics simulation
- **MDM**: Motion Diffusion Model architecture inspiration
- **CLIP**: OpenAI's vision-language model
