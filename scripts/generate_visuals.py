"""Generate recruiter-facing PNG assets from real project outputs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR  # noqa: E402
from src.recommender import DriverRecommender  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "docs" / "screenshots"
NAVY = "#17324D"
BLUE = "#2E86AB"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D95D39"
LIGHT = "#F4F7FA"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / name,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def create_model_metrics() -> None:
    with (REPORTS_DIR / "best_model_evaluation.json").open(
        encoding="utf-8"
    ) as handle:
        report = json.load(handle)
    metrics = report["metrics"]
    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Precision@5",
        "NDCG@5",
        "Top-5 success",
    ]
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "precision_at_5",
        "ndcg_at_5",
        "top_5_success_rate",
    ]
    values = [metrics[key] for key in keys]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = [BLUE] * 5 + [TEAL, TEAL, ORANGE]
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        "Held-out Test Performance — Random Forest",
        fontsize=16,
        weight="bold",
        color=NAVY,
        pad=16,
    )
    ax.text(
        0.5,
        1.01,
        "1,341 rankable candidates from 150 untouched shipments",
        transform=ax.transAxes,
        ha="center",
        color="#52606D",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", rotation=24)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
            weight="bold",
        )
    fig.tight_layout()
    _save(fig, "model_metrics.png")


def create_confusion_matrix() -> None:
    source = REPORTS_DIR / "best_model_test_confusion_matrix.png"
    if not source.exists():
        raise FileNotFoundError(
            "Run python src/evaluate_model.py before generating visuals."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, OUTPUT_DIR / "confusion_matrix.png")


def create_feature_importance() -> None:
    frame = pd.read_csv(REPORTS_DIR / "feature_importance.csv").head(10)
    frame = frame.sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6.2))
    bars = ax.barh(frame["feature"], frame["importance"], color=BLUE)
    ax.set_xlabel("Random Forest importance")
    ax.set_title(
        "Top Model Features",
        fontsize=16,
        weight="bold",
        color=NAVY,
        pad=14,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, frame["importance"]):
        ax.text(
            value + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def create_recommendation_example() -> None:
    shipment = {
        "origin_city": "Kuala Lumpur",
        "destination_city": "Penang",
        "distance_km": 355.0,
        "shipment_weight_kg": 500.0,
        "required_vehicle_type": "Van",
        "delivery_urgency": "express",
        "offered_price": 650.0,
        "pickup_hour": 9,
        "shipment_category": "retail",
    }
    results = DriverRecommender().recommend(shipment, top_k=5)
    table = pd.DataFrame(results)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["Score"] = table["predicted_match_score"].map(lambda value: f"{value:.3f}")
    table["Rating"] = table["driver_rating"].map(lambda value: f"{value:.2f}")
    table["Pickup km"] = table["pickup_distance_km"].map(
        lambda value: f"{value:.1f}"
    )
    display = table[
        [
            "Rank",
            "driver_id",
            "Score",
            "Rating",
            "availability_status",
            "current_city",
            "Pickup km",
        ]
    ]
    display.columns = [
        "Rank",
        "Driver",
        "Match score",
        "Rating",
        "Status",
        "Current city",
        "Pickup km",
    ]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    ax.set_title(
        "Example Top-5 Driver Recommendations",
        fontsize=17,
        weight="bold",
        color=NAVY,
        pad=18,
    )
    ax.text(
        0.5,
        0.94,
        "Kuala Lumpur → Penang | 500 kg | Van | Express | MYR 650",
        transform=ax.transAxes,
        ha="center",
        color="#52606D",
        fontsize=10,
    )
    rendered = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.12, 0.96, 0.7],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9.5)
    for (row, _), cell in rendered.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(LIGHT if row % 2 else "#E7F1F7")
    ax.text(
        0.02,
        0.03,
        "Only non-offline drivers with the exact vehicle type and sufficient capacity are scored.",
        transform=ax.transAxes,
        fontsize=9,
        color="#52606D",
    )
    _save(fig, "recommendation_example.png")


def create_project_workflow() -> None:
    steps = [
        ("1", "Generate\nsynthetic data", BLUE),
        ("2", "Store CSV\nand SQLite", TEAL),
        ("3", "SQL join and\nfeature engineering", ORANGE),
        ("4", "Group train /\nvalidation / test", RED),
        ("5", "Train and\nselect model", BLUE),
        ("6", "Evaluate ranking\nand classification", TEAL),
        ("7", "Serve top-5 via\nAPI and Streamlit", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(
        "End-to-End Project Workflow",
        fontsize=18,
        weight="bold",
        color=NAVY,
        pad=20,
    )
    box_width = 2.15
    gap = 0.34
    start_x = 0.28
    for index, (number, label, color) in enumerate(steps):
        x = start_x + index * (box_width + gap)
        box = FancyBboxPatch(
            (x, 1.25),
            box_width,
            1.45,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(box)
        ax.text(
            x + 0.2,
            2.38,
            number,
            color="white",
            weight="bold",
            fontsize=12,
        )
        ax.text(
            x + box_width / 2,
            1.85,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            weight="bold",
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + box_width + gap - 0.05, 1.98),
                xytext=(x + box_width + 0.05, 1.98),
                arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.8},
            )
    ax.text(
        7,
        0.65,
        "Hard rules screen vehicle, capacity, and offline status before ML ranking.",
        ha="center",
        color="#52606D",
        fontsize=10,
    )
    _save(fig, "project_workflow.png")


def main() -> None:
    create_model_metrics()
    create_confusion_matrix()
    create_feature_importance()
    create_recommendation_example()
    create_project_workflow()
    print(f"Created five visuals in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
