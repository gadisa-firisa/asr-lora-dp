import argparse
import os
import time
import random
import csv
from typing import List, Dict, Any
import torch
from utils import cfg, custom_wer, stream_data, sample_data, get_features
from model import get_model_and_processor, save_checkpoint, apply_dp, generate_pseudo_and_conf


def evaluate_on_stream(model, processor, dev_stream_gen, max_examples: int = 200) -> float:
    model.eval()
    wers = []
    with torch.no_grad():
        for i, ex in enumerate(dev_stream_gen):
            if i >= max_examples:
                break
            feats = get_features(ex, processor).unsqueeze(0).to(cfg.device)
            preds = model.generate(feats, max_length=225, num_beams=1)
            pred_text = processor.batch_decode(preds, skip_special_tokens=True)[0]
            ref = ex.get("text", "")
            wers.append(custom_wer(ref, pred_text))
    model.train()
    if len(wers) == 0:
        return float("nan")
    return sum(wers) / len(wers)

def decision_experiment_once(model, processor, noise_multiplier: float, clip_norm: float, debug_steps: int = 200) -> Dict[str, Any]:
    stream = stream_data(cfg.dataset_split)
    unlabeled_stream = sample_data(stream, cfg.unlabeled_num)
    dev_stream = sample_data(stream_data(cfg.dataset_split), cfg.dev_num)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    update_count = 0
    pseudo_kept = 0
    start_time = time.time()
    grad_norms = []

    for ex in unlabeled_stream:
        if update_count >= cfg.max_updates or update_count >= debug_steps:
            break
        feats = get_features(ex, processor)
        text, conf = generate_pseudo_and_conf(model, processor, feats)
        if conf < cfg.conf_threshold:
            continue
        tokenized = processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        labels = tokenized.input_ids.to(cfg.device)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        input_batch = feats.unsqueeze(0).to(cfg.device)
        outputs = model(input_batch, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm_before = apply_dp(model, max_grad_norm=clip_norm, noise_multiplier=noise_multiplier)
        optimizer.step()
        optimizer.zero_grad()

        update_count += 1
        pseudo_kept += 1
        grad_norms.append(grad_norm_before)

        if update_count % 50 == 0:
            dev_wer_sample = evaluate_on_stream(model, processor, sample_data(stream_data(cfg.dataset_split), cfg.dev_num), max_examples=50)
            print(f"[iter] updates={update_count} kept={pseudo_kept} loss={loss.item():.4f} gnorm={grad_norm_before:.3f} dev_sample_wer={dev_wer_sample:.3f}")

    elapsed = time.time() - start_time
    final_wer = evaluate_on_stream(model, processor, sample_data(stream_data(cfg.dataset_split), cfg.dev_num), max_examples=200)
    stats = {
        "updates": update_count,
        "pseudo_kept": pseudo_kept,
        "final_wer": final_wer,
        "mean_grad_norm": float(sum(grad_norms)/len(grad_norms)) if grad_norms else 0.0,
        "elapsed_sec": elapsed
    }
    return stats, optimizer

def run_sweep(ranks: List[int], noises: List[float], seeds: int, clip_norm: float, debug_steps: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    header = ["run_name", "rank", "noise", "seed", "updates", "pseudo_kept", "final_wer", "mean_grad_norm", "elapsed_sec", "checkpoint"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for rank in ranks:
        for noise in noises:
            for seed in range(seeds):
                run_name = f"r{rank}_n{noise}_s{seed}_{int(time.time())}"
                print("\n=== RUN:", run_name, "===\n")
                torch.manual_seed(cfg.seed + seed)
                random.seed(cfg.seed + seed)
                model, processor = get_model_and_processor(cfg.model_name, lora_rank=rank)
                stats, optimizer = decision_experiment_once(model, processor, noise_multiplier=noise, clip_norm=clip_norm, debug_steps=debug_steps)
                ckpt_path = save_checkpoint(model, optimizer, out_dir, run_name)
                row = [
                    run_name, rank, noise, seed, stats["updates"], stats["pseudo_kept"],
                    stats["final_wer"], stats["mean_grad_norm"], stats["elapsed_sec"], ckpt_path
                ]
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print("Logged run ->", csv_path)

    return csv_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=cfg.model_name)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4,16])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.0,1.0,2.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--debug_steps", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default=cfg.out_dir)
    args = parser.parse_args()

    cfg.model_name = args.model
    cfg.out_dir = args.out_dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_file = run_sweep(ranks=args.ranks, noises=args.noises, seeds=args.seeds, clip_norm=args.clip_norm, debug_steps=args.debug_steps, out_dir=cfg.out_dir)
    
