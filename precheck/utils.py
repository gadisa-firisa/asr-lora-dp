from dataclasses import dataclass
from typing import Generator, Dict, Any
import torch
from datasets import load_dataset


@dataclass
class Config:
    model_name: str = "openai/whisper-tiny"
    dataset_name: str = "openslr/librispeech_asr"
    dataset_config_name: str = "clean"
    dataset_split: str = "train.100"
    unlabeled_num: int = 3000
    dev_num: int = 500
    max_updates: int = 200            
    conf_threshold: float = -0.2
    lr: float = 1e-4
    batch_size: int = 1
    seed: int = 12345
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "sweep_out"

cfg = Config()

def custom_wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    d = [[0] * (len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)] / max(1, len(r))


def stream_data(split: str) -> Generator[Dict[str, Any], None, None]:
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config_name, split=split, streaming=True)
    return (ex for ex in ds)

def sample_data(dataset_generator: Generator[Dict[str, Any], None, None], n: int):
    i = 0
    for ex in dataset_generator:
        yield ex
        i += 1
        if i >= n:
            break

def get_features(example: Dict[str, Any], processor):
    audio = example["audio"]
    arr = audio["array"]
    sampling_rate = audio.get("sampling_rate", 16000)
    inputs = processor.feature_extractor(arr, sampling_rate=sampling_rate, return_tensors="pt")
    return inputs.input_features[0]