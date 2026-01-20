from typing import Dict, Any, Generator
import torch
from datasets import load_dataset
from asr_federated.task import Cfg, get_model_and_processor, custom_wer, get_features
from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from asr_federated.dp import DPAccountant

client_app = ClientApp()


@client_app.train()
def train(message: Message, context: Context) -> Message:
    rd = message.content
    config_record = rd.get("config")
    cfg = dict(config_record) if config_record is not None else dict(context.run_config or {})
    
    lora_rank = int(cfg.get("lora_rank", cfg.get("lora-rank", Cfg.lora_rank)))
    
    model, processor = get_model_and_processor(Cfg.model_name, lora_rank)
    model = model.to(Cfg.device)
    model.train()

    arrays_record = rd.get("arrays")

    if arrays_record is not None:
        server_state = arrays_record.to_torch_state_dict()
        local_sd = model.state_dict()
        for k, v in server_state.items():
            if k in local_sd:
                local_sd[k].copy_(v.to(local_sd[k].device))
        model.load_state_dict(local_sd, strict=False)

    local_updates = int(cfg.get("local_updates", Cfg.local_updates))
    clip_norm = float(cfg.get("clip_norm", Cfg.clip_norm))
    noise_multiplier = float(cfg.get("noise_multiplier", Cfg.noise_multiplier))
    conf_threshold = float(cfg.get("conf_threshold", Cfg.conf_threshold))

    node_id = int(context.node_id or 0)
    stream = load_dataset(Cfg.dataset_name, Cfg.dataset_config, split=Cfg.dataset_split, streaming=True)
    start_skip = node_id * Cfg.unlabeled_per_client
    unlabeled_gen = (ex for i, ex in enumerate(stream) if i >= start_skip)
    unlabeled_gen = (next(unlabeled_gen) for _ in [] )

    def client_chunk(split_start: int, count: int) -> Generator[Dict[str, Any], None, None]:
        ds_gen = load_dataset(Cfg.dataset_name, Cfg.dataset_config, split=Cfg.dataset_split, streaming=True)
        i = 0
        yielded = 0
        for ex in ds_gen:
            if i < split_start:
                i += 1
                continue
            if yielded >= count:
                break
            yield ex
            i += 1
            yielded += 1

    unlabeled_gen = client_chunk(start_skip, Cfg.unlabeled_per_client)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Cfg.lr)

    sample_rate = Cfg.batch_size / max(1, Cfg.unlabeled_per_client)

    dp_accountant = DPAccountant(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        delta=1e-5,
        max_grad_norm=clip_norm,
    )

    updates = 0
    total_examples = 0
    running_loss = 0.0
    grad_norms = []

    for ex in unlabeled_gen:
        if updates >= local_updates:
            break

        feats = get_features(ex, processor).to(Cfg.device).unsqueeze(0)  # batch dim
        model.eval()

        with torch.no_grad():
            gen_ids = model.generate(feats, max_length=225, num_beams=1)
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            tokenized = processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            labels = tokenized.input_ids.to(Cfg.device)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            out = model(feats, labels=labels)

            gen_loss = out.loss.item()
            conf = -gen_loss / max(1, (labels != -100).sum().item())

        model.train()

        if conf < conf_threshold:
            continue

        outputs = model(feats, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
        
        if len(grads) > 0:
            total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
            clip_coef = (clip_norm / (total_norm + 1e-6)).clamp(max=1.0)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.detach().mul_(clip_coef.to(p.grad.device))

            if noise_multiplier > 0.0:
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        noise_std = noise_multiplier * clip_norm
                        noise = torch.normal(mean=0.0, std=noise_std, size=p.grad.shape, device=p.grad.device)
                        p.grad.add_(noise)

            grad_norms.append(total_norm.item())

        else:
            grad_norms.append(0.0)

        optimizer.step()
        
        dp_accountant.step()
        
        updates += 1
        total_examples += 1
        running_loss += float(loss.item())

    privacy_spent = dp_accountant.get_privacy_spent()
    epsilon = privacy_spent["epsilon"]
    
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    arrays_reply = ArrayRecord(lora_state)

    metrics = MetricRecord({
        "loss": running_loss / max(1, total_examples),
        "num-examples": total_examples,
        "mean_grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
        "epsilon": epsilon,
        "delta": privacy_spent["delta"],
        "dp_steps": privacy_spent["steps"],
    })

    out_rd = RecordDict({"arrays": arrays_reply, "metrics": metrics})
    
    print(f"[Client {node_id}] Finished {updates} updates, ε={epsilon:.4f}")
    return Message(content=out_rd)


@client_app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    config_record = message.content.get("config")
    cfg = dict(config_record) if config_record is not None else dict(context.run_config or {})
    lora_rank = int(cfg.get("lora_rank", cfg.get("lora-rank", Cfg.lora_rank)))
    
    model, processor = get_model_and_processor(Cfg.model_name, lora_rank)
    model = model.to(Cfg.device)
    model.eval()

    arrays_record = message.content.get("arrays")
    if arrays_record is not None:
        server_state = arrays_record.to_torch_state_dict()
        local_sd = model.state_dict()
        
        for k, v in server_state.items():
            if k in local_sd:
                local_sd[k].copy_(v.to(local_sd[k].device))

        model.load_state_dict(local_sd, strict=False)

    node_id = int(context.node_id or 0)
    start_skip = node_id * Cfg.dev_per_client
    ds_gen = load_dataset(Cfg.dataset_name, Cfg.dataset_config, split=Cfg.eval_split, streaming=True)
    
    i = 0
    taken = 0
    wers = []
    for ex in ds_gen:
        if i < start_skip:
            i += 1
            continue
        if taken >= Cfg.dev_per_client:
            break
        feats = get_features(ex, processor).unsqueeze(0).to(Cfg.device)

        with torch.no_grad():
            pred_ids = model.generate(feats, max_length=225, num_beams=1)
            pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        ref = ex.get("text", "")
        wers.append(custom_wer(ref, pred))
        taken += 1
        i += 1

    mean_wer = sum(wers) / len(wers) if wers else float("nan")
    metrics = MetricRecord({"wer": mean_wer, "num-examples": len(wers)})
    out_rd = RecordDict({"metrics": metrics})

    return Message(content=out_rd)
