## Overview
This repository experiments effects of DP and PEFT on federated ASR learning.

## Repository Structure
```bash

asr-lora-dp/
├── asr_federated/
│   ├── __init__.py
│   ├── client.py
│   ├── dp.py
│   ├── logging.py
│   ├── server.py
│   └── task.py
├── precheck/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── plot.py
│   └── utils.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt

```

### 1. Precheck Experiment:
This module is runs decision experiment for speech recognition under constraints using Whisper.

The goal of the experiment is to check whether Whisper can still learn anything meaningful when pseudo-labeling, parameter-efficient fine-tuning (LoRA), and DP-style gradient noise are combined. 

---


Privacy-preserving on-device ASR could be generally assumed to be achievable by using techniques:

- **Privacy**: Differential privacy
- **Efficiency**: LoRA adapters
- **No labels**: Pseudo-labeling
- **Scale**: Federated learning



**Q**: Could it be the case that noise introduced for privacy destroys the already-weak learning signal produced by pseudo-labels and compressed by LoRA?.

---

## Hypothesis

There exists a noise limit beyond which:

- pseudo-labeling errors,
- gradient clipping,
- DP noise,
- and low-rank adaptation

combined cause compounding errors, leading to stalled learning, divergence, or hallucinations in encoder–decoder ASR models such as Whisper.

---

## Experimental setup

### Model and Dataset

- `openai/whisper-tiny` (or `whisper-small`)
- LoRA applied to attention projections (`q_proj`, `v_proj`)
- All base Whisper weights frozen
- LibriSpeech dataset 


---

## Training loop

For each audio sample:

- **Pseudo-label generation**
   - The Whisper model transcribes the audio

- **Confidence filtering**
   - Per-token negative log-likelihood is computed
   - Low-confidence pseudo-labels are discarded

- **Constrained update**
   - Loss is computed on the pseudo-label
   - Only LoRA parameters receive gradients
   - Gradients are clipped
   - Gaussian noise is added (DP simulation)

- **Parameter update**
   - A single optimizer step is applied



---

## Metrics tracked:


| Metric | Meaning |
|------|--------|
| `final_wer` | Can the model still transcribe speech? |
| `pseudo_kept` | How many pseudo-labels survive confidence filtering |
| `grad_norm` | Signal strength before clipping/noise |
| `steps_to_collapse` | When learning stalls or reverses |


---

## How to run the experiment

```bash
cd precheck

python3 main.py \
  --model openai/whisper-tiny \
  --ranks 4 16 \
  --noises 0.0 1.0 2.0 \
  --debug_steps 200
```

Outputs:

- CSV file with per-run metrics
- Heatmap of WER vs. (LoRA rank, noise level)
- LoRA checkpoints

---

## Interpreting results

| Observation | Interpretation |
|------------|---------------|
| WER improves | Learning survives constraints |
| WER flat | Signal ≈ noise |
| WER degrades | Error compounding |
| `pseudo_kept → 0` | Confidence collapse |


If Whisper cannot learn under these constraints centrally, it will not be useful to run the federated learning experiment.

---

## 2. Federated Learning Experiment

The directory `asr_federated/` contains **federated learning simulation** with:

- **Per-round CSV logging**: Track WER, loss, and ε across rounds
- **Differential privacy accounting**: RDP-based ε estimation 
- **Sweep mode**: Run hyperparameter sweeps and generate heatmaps
with the assumption that the precheck experiment has shown that the model can learn under the given constraints.

### Running with `uv`
#### Install uv: https://docs.astral.sh/uv/getting-started/installation/
```bash
# Run with default config (3 clients, 3 rounds)
uv run flwr run . 

# Override config at runtime
uv run flwr run . --run-config "num-rounds=10 lora-rank=4 noise-multiplier=0.5"

# Sweep mode
uv run flwr run . --run-config "num-rounds=5 lora-rank=4 noise-multiplier=0.5 sweep-mode=true"
uv flwr run . --run-config "num-rounds=5 lora-rank=8 noise-multiplier=1.0 sweep-mode=true"
```

### Running with `flwr run`

#### Installation

```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -e .
```


```bash
# Run with default config (3 clients, 3 rounds)
flwr run . 

# Override config at runtime
flwr run . --run-config "num-rounds=10 lora-rank=4 noise-multiplier=0.5"

# Sweep mode
flwr run . --run-config "num-rounds=5 lora-rank=4 noise-multiplier=0.5 sweep-mode=true"
flwr run . --run-config "num-rounds=5 lora-rank=8 noise-multiplier=1.0 sweep-mode=true"
```

### Configuration (pyproject.toml)

| Key | Default | Description |
|-----|---------|-------------|
| `num-rounds` | 3 | Number of FL rounds |
| `local-updates` | 20 | DP-SGD steps per client per round |
| `lora-rank` | 8 | LoRA rank |
| `noise-multiplier` | 1.0 | DP noise σ/C ratio |
| `clip-norm` | 1.0 | Gradient clipping norm |
| `log-dir` | logs | Output directory for CSVs |
| `sweep-mode` | false | Enable sweep mode |

The number of clients can be configured in `[tool.flwr.federations.local-simulation]` as `options.num-supernodes`.

### Output Files

- `logs/<run_name>_rounds.csv`: Per-round metrics (WER, loss, ε, timestamp)
- `logs/sweep_results.csv`: Aggregated sweep results
- `logs/heatmap_final_wer.png`: WER vs noise × rank heatmap
- `final_lora_state_<timestamp>.pt`: Final LoRA weights


## Interpreting results

| Observation | Interpretation |
|------------|----------------|
| WER improves | Learning survives constraints |
| WER flat | Signal ≈ noise |
| WER degrades | Error compounding |
| `pseudo_kept → 0` | Confidence collapse |
| `ε → ∞` | No privacy (noise=0) |
| `ε < 10` | Reasonable privacy budget |

---
