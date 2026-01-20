import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

class Cfg:
    model_name = "openai/whisper-tiny"
    dataset_name = "openslr/librispeech_asr"
    dataset_config = "clean"
    dataset_split = "train.100" 
    eval_split = "validation.clean"
    unlabeled_per_client = 500 
    dev_per_client = 100
    lora_rank = 8
    lr = 1e-4
    batch_size = 1
    conf_threshold = -0.2
    clip_norm = 1.0
    noise_multiplier = 1.0 
    local_updates = 20 
    seed = 12345
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = "logs"
    sweep_mode = False


def custom_wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(r)][len(h)] / max(1, len(r))


def get_features(example, processor: WhisperProcessor):
    audio = example["audio"]
    arr = audio["array"]
    sr = audio.get("sampling_rate", 16000)
    inp = processor.feature_extractor(arr, sampling_rate=sr, return_tensors="pt")
    return inp.input_features[0]


def get_model_and_processor(model_name: str, lora_rank: int):
    
    processor = WhisperProcessor.from_pretrained(model_name)
    base = WhisperForConditionalGeneration.from_pretrained(model_name)
   
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(base, lora_cfg)
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False

    return model, processor
