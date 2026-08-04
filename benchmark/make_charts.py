"""Task 6 — penalty vs n, runtime vs n charts."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

GROUP_COLOR = {"small": "#2563eb", "medium": "#f59e0b", "stress": "#dc2626"}
GROUP_MARKER = {"small": "o", "medium": "s", "stress": "^"}


def main():
    with open(os.path.join(HERE, "results", "results.json")) as f:
        rows = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    for group in ["small", "medium", "stress"]:
        pts = [(r["n"], r["penalty"]) for r in rows if r["group"] == group and r["feasible"]]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, color=GROUP_COLOR[group], marker=GROUP_MARKER[group],
                   label=f"{group} (feasible)", s=70, zorder=3)
    infeasible = [(r["n"], r["label"]) for r in rows if not r["feasible"]]
    for n, label in infeasible:
        ax.axvline(n, color="#dc2626", alpha=0.15, linestyle="--", zorder=1)
    ax.set_xlabel("n (number of tasks)")
    ax.set_ylabel("P(sigma) — penalty")
    ax.set_title("Penalty vs n (dashed red lines = infeasible instances, no penalty defined)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "charts", "penalty_vs_n.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for group in ["small", "medium", "stress"]:
        pts = [(r["n"], r["runtime_ms"]) for r in rows if r["group"] == group]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, color=GROUP_COLOR[group], marker=GROUP_MARKER[group],
                label=group, linewidth=1.5, markersize=8)
    ax.set_xlabel("n (number of tasks)")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("WD-VTR runtime vs n (all 9 instances, feasible + infeasible)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "charts", "runtime_vs_n.png"), dpi=150)
    plt.close(fig)

    print("Wrote charts/penalty_vs_n.png and charts/runtime_vs_n.png")


if __name__ == "__main__":
    main()
