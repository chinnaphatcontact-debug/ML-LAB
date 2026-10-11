import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    OUTPUT_DIR / "architecture_comparison.json",
    "r",
    encoding="utf-8"
) as file:
    results = json.load(file)


models = list(results.keys())

test_accuracy = [
    results[model]["test_accuracy"] * 100
    for model in models
]

val_accuracy = [
    results[model]["best_val_accuracy"] * 100
    for model in models
]

parameters = [
    results[model]["total_params"]
    for model in models
]


# ============================================================
# 1. TEST ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    models,
    test_accuracy
)

plt.title(
    "CNN Architecture Comparison - Test Accuracy",
    fontsize=16
)

plt.xlabel(
    "CNN Architecture",
    fontsize=12
)

plt.ylabel(
    "Test Accuracy (%)",
    fontsize=12
)

plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    test_accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "architecture_test_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    models,
    val_accuracy
)

plt.title(
    "CNN Architecture Comparison - Best Validation Accuracy",
    fontsize=16
)

plt.xlabel(
    "CNN Architecture",
    fontsize=12
)

plt.ylabel(
    "Best Validation Accuracy (%)",
    fontsize=12
)

plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    val_accuracy
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "architecture_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. PARAMETERS
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    models,
    parameters
)

plt.title(
    "CNN Architecture Comparison - Trainable Parameters",
    fontsize=16
)

plt.xlabel(
    "CNN Architecture",
    fontsize=12
)

plt.ylabel(
    "Number of Parameters",
    fontsize=12
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    parameters
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(parameters) * 0.02,
        f"{value:,}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "architecture_parameters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("ARCHITECTURE COMPARISON PLOTS")
print("=" * 70)

print(
    OUTPUT_DIR /
    "architecture_test_accuracy.png"
)

print(
    OUTPUT_DIR /
    "architecture_validation_accuracy.png"
)

print(
    OUTPUT_DIR /
    "architecture_parameters.png"
)

print("=" * 70)