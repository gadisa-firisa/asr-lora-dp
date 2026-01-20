import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_heatmap(csv_path: str, out_dir: str, metric: str = "final_wer"):

    df = pd.read_csv(csv_path)
    pivot = df.groupby(["rank", "noise"])[metric].mean().reset_index()
    ranks = sorted(pivot["rank"].unique())
    noises = sorted(pivot["noise"].unique())
    mat = []

    for r in ranks:
        row = []
        for n in noises:
            cell = pivot[(pivot["rank"] == r) & (pivot["noise"] == n)][metric]
            val = float(cell.values[0]) if len(cell) > 0 else float("nan")
            row.append(val)
        mat.append(row)

    plt.figure(figsize=(4 + len(noises)*0.5, 3 + len(ranks)*0.3))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(noises)), [str(x) for x in noises])
    plt.yticks(range(len(ranks)), [str(x) for x in ranks])
    plt.xlabel("noise_multiplier")
    plt.ylabel("lora_rank")
    plt.title(f"Heatmap ({metric})")

    out_path = os.path.join(out_dir, f"heatmap_{metric}.png")
    plt.savefig(out_path, bbox_inches="tight")
    print("Saved heatmap ->", out_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="final_wer")
    args = parser.parse_args()
    plot_heatmap(args.csv_path, args.out_dir, args.metric)
    