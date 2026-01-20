from utils import cfg
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model


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
    model.to(cfg.device)
    return model, processor

def save_checkpoint(model, optimizer, out_dir, run_name):
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "lora_state_dict": {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path = os.path.join(out_dir, f"{run_name}.pt")
    torch.save(state, path)
    return path

def apply_dp(model: torch.nn.Module, max_grad_norm: float, noise_multiplier: float) -> float:
    
    grads = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
    if len(grads) == 0:
        return 0.0
    total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    clip_coef = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))

    if noise_multiplier > 0.0:
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                noise_std = noise_multiplier * max_grad_norm
                noise = torch.normal(mean=0.0, std=noise_std, size=p.grad.shape, device=p.grad.device, dtype=p.grad.dtype)
                p.grad.add_(noise)

    return total_norm.item()

@torch.no_grad()
def generate_pseudo_and_conf(model, processor, feats):
    model.eval()
    input_b = feats.unsqueeze(0).to(cfg.device)
    gen_ids = model.generate(input_b, max_length=225, num_beams=1)
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    
    tokenized = processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    labels = tokenized.input_ids.to(cfg.device)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    outputs = model(input_b, labels=labels)
    loss = outputs.loss.item()
    conf = -loss / max(1, (labels != -100).sum().item())
    model.train()
    
    return text, conf