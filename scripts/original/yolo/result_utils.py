from pathlib import Path

import matplotlib.pyplot as plt


def save_test_results(metrics, save_dir=None):
    """Write a per-class results.csv and results.png (mAP50 / mAP50-95 bar chart)
    for a single model.val() call, mirroring what model.train() saves per-epoch.
    """
    save_dir = Path(save_dir) if save_dir is not None else Path(metrics.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "results.csv").write_text(metrics.to_csv())

    summary = metrics.summary()
    classes = [row["Class"] for row in summary]
    map50 = [row["mAP50"] for row in summary]
    map5095 = [row["mAP50-95"] for row in summary]

    x = range(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 6))
    ax.bar([i - width / 2 for i in x], map50, width, label="mAP50")
    ax.bar([i + width / 2 for i in x], map5095, width, label="mAP50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("mAP")
    ax.set_title("Per-class mAP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close(fig)

    return save_dir / "results.csv", save_dir / "results.png"
