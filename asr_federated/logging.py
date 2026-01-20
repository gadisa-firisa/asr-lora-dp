import os
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FederatedLogger:
    
    def __init__(
        self,
        out_dir: str = "logs",
        run_name: Optional[str] = None,
        extra_columns: Optional[List[str]] = None,
    ):
        self.extra_columns = extra_columns or []
        self.out_dir = out_dir
        self.run_name = run_name or f"run_{int(time.time())}"
        os.makedirs(out_dir, exist_ok=True)
        
        self.columns = [
            "round",
            "timestamp",
            "mean_wer",
            "mean_loss",
            "epsilon",
            "num_clients",
            "total_examples",
        ]
        
        if extra_columns:
            self.columns.extend(extra_columns)
        
        self.csv_path = os.path.join(out_dir, f"{self.run_name}_rounds.csv")
        self._file = None
        self._writer = None
        self._initialized = False
        self._rounds_logged = 0
    
    def _init_csv(self, extra_keys: List[str] = None) -> None:
        if self._initialized:
            return
        
        if extra_keys:
            for key in extra_keys:
                if key not in self.columns:
                    self.columns.append(key)
        
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self._initialized = True
    
    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, Any],
        num_clients: int = 0,
        total_examples: int = 0,
    ) -> None:
        
        if not self._initialized:
            extra_keys = [k for k in metrics.keys() if k not in self.columns]
            self._init_csv(extra_keys)
        
        row = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "mean_wer": metrics.get("wer", metrics.get("mean_wer", "")),
            "mean_loss": metrics.get("loss", metrics.get("mean_loss", "")),
            "epsilon": metrics.get("epsilon", ""),
            "num_clients": num_clients,
            "total_examples": total_examples,
        }
        
        for key, value in metrics.items():
            if key not in row:
                row[key] = value
        
        self._writer.writerow(row)
        self._file.flush()
        self._rounds_logged += 1
        
        print(f"Round {round_num}: WER={row['mean_wer']:.4f}, "
              f"Loss={row['mean_loss']:.4f}, ε={row['epsilon']}")
    
    def finalize(self) -> str:
        
        if self._file:
            self._file.close()
            self._file = None

        print(f"Logged {self._rounds_logged} rounds to {self.csv_path}")
        return self.csv_path
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


class SweepLogger:
    
    def __init__(self, out_dir: str = "sweep_logs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "sweep_results.csv")
        
        self.columns = [
            "run_name",
            "noise_multiplier",
            "lora_rank",
            "num_rounds",
            "final_wer",
            "final_loss",
            "final_epsilon",
            "timestamp",
        ]
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
    
    def log_run(
        self,
        run_name: str,
        noise_multiplier: float,
        lora_rank: int,
        num_rounds: int,
        final_metrics: Dict[str, Any],
    ) -> None:

        row = {
            "run_name": run_name,
            "noise_multiplier": noise_multiplier,
            "lora_rank": lora_rank,
            "num_rounds": num_rounds,
            "final_wer": final_metrics.get("wer", ""),
            "final_loss": final_metrics.get("loss", ""),
            "final_epsilon": final_metrics.get("epsilon", ""),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writerow(row)
        
        print(f"Logged {run_name}: noise={noise_multiplier}, rank={lora_rank}")
    
    def generate_heatmap(
        self,
        metric: str = "final_wer",
        x_col: str = "noise_multiplier",
        y_col: str = "lora_rank",
        output_name: Optional[str] = None,
    ) -> Optional[str]:
       
        return plot_heatmap(
            csv_path=self.csv_path,
            x_col=x_col,
            y_col=y_col,
            value_col=metric,
            out_path=os.path.join(self.out_dir, f"{output_name or f'heatmap_{metric}'}.png"),
        )


def plot_heatmap(
    csv_path: str,
    x_col: str,
    y_col: str,
    value_col: str,
    out_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
) -> Optional[str]:
    
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("Empty CSV, skipping heatmap")
        return None
    
    pivot = df.groupby([y_col, x_col])[value_col].mean().reset_index()
    
    x_vals = sorted(pivot[x_col].unique())
    y_vals = sorted(pivot[y_col].unique())
    
    matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, row in pivot.iterrows():
        xi = x_vals.index(row[x_col])
        yi = y_vals.index(row[y_col])
        matrix[yi, xi] = row[value_col]
    
    fig, ax = plt.subplots(figsize=(4 + len(x_vals) * 0.6, 3 + len(y_vals) * 0.4))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{x:.2g}" for x in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(y) for y in y_vals])
    
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title or f"{value_col.replace('_', ' ').title()} Heatmap")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col.replace("_", " ").title())
    
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmean(matrix) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved heatmap to {out_path}")
    return out_path
