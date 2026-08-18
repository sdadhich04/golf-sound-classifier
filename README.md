# Golf Swing Sound Classifier — TinyML on Arduino Nano 33 BLE Sense

**EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing | University of Washington, Spring 2026**

Classifying golf swing outcomes in real time from a single contact-sound recording, running entirely on a microcontroller with no cloud dependency.

---

## What it does

The system listens for the ~500 ms window around a club-ball impact and classifies the shot into one of three categories:

| Class | Description |
|-------|-------------|
| **Good** | Clean center-face strike |
| **Ground** | Club hits turf before the ball |
| **Top** | Club catches the top of the ball |

Audio is captured by the onboard PDM microphone on an **Arduino Nano 33 BLE Sense**, preprocessed into Mel Filterbank Energy (MFE) features, and fed to a quantized dense neural network — all running on a microcontroller with 256 KB flash and 64 KB RAM.

---

## Results

Three Edge Impulse impulses were developed and benchmarked:

| Model | Window | Val Accuracy | AUC | Notes |
|-------|--------|-------------|-----|-------|
| Baseline (#1) | 500 ms | 74.5% | — | Initial model |
| Compressed (#2) | 250 ms | 54.8% | — | Smallest footprint |
| **Final (#3)** | **500 ms** | **91.6%** | **0.98** | Deployed model |

The offline notebook evaluates the same approach on a strict held-out test set (original clips only, no augmentation leakage) and achieves **79.4% test accuracy** — a conservative lower bound compared to the Edge Impulse validation split. The unoptimized float32 model scores 78.72% on the same EI test set, confirming the offline notebook faithfully reproduces the approach.

Per-class performance (Final model, int8 quantized, validation set):

| Class | Accuracy | F1 |
|-------|----------|----|
| Good | 96.2% | 0.92 |
| Ground | 86.7% | 0.90 |
| Top | 88.0% | 0.94 |

Ground → Good misclassification (13.3%) is the primary failure mode, likely due to acoustic similarity when the divot is small.

---

## On-device performance (Arduino Nano 33 BLE Sense, EON Compiler)

| | MFE extraction | Classifier | **Total** |
|---|---|---|---|
| **Latency** | 125 ms | 2 ms | **127 ms** |
| **Peak RAM** | 11.8 KB | 3.3 KB | **11.8 KB** |
| **Flash** | — | 75.9 KB | **75.9 KB** |

> **Why int8 quantization is required, not optional:** the unoptimized float32 model uses 258 KB flash — exceeding the Nano's 256 KB limit. Int8 quantization shrinks the classifier to 75.9 KB and cuts RAM from 9.0 KB to 3.3 KB, with no measurable accuracy loss.

---

## Repository contents

```
Golf_TinyML_Pipeline.ipynb          ← Full offline pipeline (data → features → train → quantize → evaluate)
demo_live_inference.mp4             ← Live demo: classifier running on Arduino Nano 33 BLE Sense
GolfSoundClassifier_EE446_FinalProjectReport.pdf   ← Written report
Golf_TinyML_EE446.pptx             ← Final presentation slides
project_proposal.pdf               ← Original project proposal
deployment/                         ← Edge Impulse C++ library export (EON Compiler, int8 quantized)
  golf-cpp-mcu-v2-impulse-3.zip     ← Arduino-ready deployment package for Nano 33 BLE Sense
dataset/                            ← Raw recording sessions (MP3, ~13 MB total)
  20260528_125335.mp3               ← Recording session 1 (May 28 2026)
  20260528_130155.mp3               ← Recording session 2
  20260528_131029.mp3               ← Recording session 3
```

---

## Model architecture

**Deployed model (Edge Impulse, Impulse #3)** — fully connected network on MFE features:

```
Input  (1,960 features — 40 MFE bins × 49 frames, 500 ms window)
Dense  (32 neurons, ReLU)
Dropout (0.25)
Dense  (16 neurons, ReLU)
Dropout (0.25)
Output (3 classes, softmax)
```

Training: 100 cycles, learning rate 0.005, CPU processor, no learned optimizer.

**Offline notebook model** — 1D CNN for methodology comparison:

```
Input  (50, 40)
Conv1D (8 filters, kernel 3, ReLU) → MaxPool(2) → Dropout(0.25)
Conv1D (16 filters, kernel 3, ReLU) → MaxPool(2) → Dropout(0.25)
Flatten → Dense (3, softmax)
```

The notebook reproduces the full MFE feature extraction and int8 quantization pipeline in plain Python/TensorFlow, independent of Edge Impulse.

---

## Pipeline overview (`Golf_TinyML_Pipeline.ipynb`)

1. **Data loading** — WAV files labeled by filename prefix (`good`, `grnd`, `top`); downmixed to mono, resampled to 16 kHz to match the board's PDM mic
2. **Stratified train/test split** — held-out test set contains only original (non-augmented) clips to avoid leakage
3. **Augmentation** — time shift (±30 ms), additive noise (SNR 30–40 dB), small gain variation; no pitch shift or time-stretch (preserves transient character of short top-shots)
4. **MFE features** — 40 Mel filterbank energies, 500 ms window, 10 ms hop; matches Edge Impulse's MFE processing block exactly
5. **1D CNN** — two conv blocks (8 → 16 filters) with max-pooling and dropout, softmax output
6. **Int8 quantization** — TFLite full-integer quantization with representative dataset calibration; input/output both int8 for maximum on-device efficiency
7. **Per-clip inference** — single function that replicates the on-device inference loop: load WAV → extract MFE → quantize input → run interpreter → dequantize output → return class + confidence

---

## Hardware

- **Arduino Nano 33 BLE Sense** (Nordic nRF52840, 256 KB flash, 64 KB RAM, onboard PDM microphone)
- Edge Impulse project: [sdadhich / Golf](https://studio.edgeimpulse.com/public/1027938/live) (public — clone to retrain or deploy)
- Offline companion notebook reproduces the full methodology in plain Python/TensorFlow

---

## Dataset

The raw recording sessions (three ~5-minute MP3 files from May 28, 2026) are included in `dataset/`. These are the original bulk recordings captured during the data collection session.

The individual labeled clips (short WAV files prefixed `good_`, `grnd_`, `top_`) were trimmed from these recordings and uploaded to Edge Impulse for data management and augmentation. They are not stored in this repo — to reproduce results with the offline notebook, export the labeled dataset from Edge Impulse and place the WAV files in a `Golf dataset/` folder at the repo root, or record and label your own swings following the same naming convention.

**Dependencies:** `numpy`, `soundfile`, `librosa`, `matplotlib`, `tensorflow`, `scikit-learn`

```bash
pip install numpy soundfile librosa matplotlib tensorflow scikit-learn
```

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience
