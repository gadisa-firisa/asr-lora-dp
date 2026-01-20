
import time
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
)
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg

from asr_federated.logging import FederatedLogger, SweepLogger
from asr_federated.task import Cfg, get_model_and_processor

server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    print("ServerApp.main() starting. Context:", context.run_id, "node:", context.node_id)
    
    cfg = context.run_config
    lora_rank = int(cfg.get("lora-rank", 8))
    local_updates = int(cfg.get("local-updates", 20))
    lr = float(cfg.get("lr", 1e-4))

    clip_norm = float(cfg.get("clip-norm", 1.0))
    noise_multiplier = float(cfg.get("noise-multiplier", 1.0))
    conf_threshold = float(cfg.get("conf-threshold", -0.2))

    log_dir = str(cfg.get("log-dir", "logs"))
    sweep_mode = bool(cfg.get("sweep-mode", False))
    num_rounds = int(cfg.get("num-rounds", 3))
    
    print(f"Config: lora_rank={lora_rank}, local_updates={local_updates}, lr={lr}")
    print(f"noise_multiplier={noise_multiplier}, clip_norm={clip_norm}, num_rounds={num_rounds}")
    
    model, _ = get_model_and_processor(Cfg.model_name, lora_rank)
    model = model.to("cpu")

    state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    arrays = ArrayRecord(state) 

    strategy = FedAvg(
        fraction_train=1.0,  
        fraction_evaluate=1.0,
        arrayrecord_key="arrays", 
        configrecord_key="config",
        weighted_by_key="num-examples",
    )

    config_rec = ConfigRecord(
        {
            "local_updates": local_updates,
            "lr": lr,
            "clip_norm": clip_norm,
            "noise_multiplier": noise_multiplier,
            "conf_threshold": conf_threshold,
            "lora_rank": lora_rank,
        }
    )
    
    run_name = f"fed_r{lora_rank}_n{noise_multiplier}_{int(time.time())}"
    logger = FederatedLogger(out_dir=log_dir, run_name=run_name)
    
    print(f"Starting FedAvg with num_rounds={num_rounds}, logging to {log_dir}/{run_name}")
    
    current_arrays = arrays
    cumulative_epsilon = 0.0
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Round {round_num}/{num_rounds} ===")
        
        result = strategy.start(
            grid=grid, 
            initial_arrays=current_arrays, 
            train_config=config_rec, 
            num_rounds=1
        )
        
        train_metrics = {}
        eval_metrics = {}
        num_clients = 0
        total_examples = 0
        round_epsilon = 0.0
        
        if hasattr(result, 'train_metrics') and result.train_metrics:
            for m in result.train_metrics:
                if 'loss' in m:
                    train_metrics['loss'] = m.get('loss', 0)
                if 'epsilon' in m:
                    round_epsilon = max(round_epsilon, m.get('epsilon', 0))
                if 'num-examples' in m:
                    total_examples += m.get('num-examples', 0)
                num_clients += 1
        
        if hasattr(result, 'eval_metrics') and result.eval_metrics:
            wers = [m.get('wer', 0) for m in result.eval_metrics if 'wer' in m]
            if wers:
                eval_metrics['wer'] = sum(wers) / len(wers)
        
        cumulative_epsilon = max(cumulative_epsilon, round_epsilon)
        
        logger.log_round(
            round_num=round_num,
            metrics={
                'wer': eval_metrics.get('wer', 0.0),
                'loss': train_metrics.get('loss', 0.0),
                'epsilon': cumulative_epsilon,
                'round_epsilon': round_epsilon,
            },
            num_clients=num_clients,
            total_examples=total_examples,
        )
        

        current_arrays = result.arrays
    
    csv_path = logger.finalize()
    print(f"\nTraining complete. Logs saved to: {csv_path}")
    

    final_state_dict = current_arrays.to_torch_state_dict()
    model_path = f"final_lora_state_{int(time.time())}.pt"
    torch.save(final_state_dict, model_path)
    print(f"Final LoRA saved to: {model_path}")
    
    if sweep_mode:
        sweep_logger = SweepLogger(out_dir=log_dir)
        sweep_logger.log_run(
            run_name=run_name,
            noise_multiplier=noise_multiplier,
            lora_rank=lora_rank,
            num_rounds=num_rounds,
            final_metrics={
                'wer': eval_metrics.get('wer', 0.0),
                'loss': train_metrics.get('loss', 0.0),
                'epsilon': cumulative_epsilon,
            },
        )

