from sklearn.datasets import load_wine
import pandas as pd


def load_dataset():
    """
    Load the Wine dataset from scikit-learn.

    Returns:
        X: Input features
        y: Target labels
        feature_names: Names of input features
        target_names: Names of target classes
    """

    wine = load_wine()

    X = wine.data
    y = wine.target

    feature_names = wine.feature_names
    target_names = wine.target_names

    return X, y, feature_names, target_names


def show_dataset_info(X, y, feature_names, target_names):
    """
    Display basic information about the dataset.
    """

    print("=" * 60)
    print("                 WINE DATASET")
    print("=" * 60)

    print(f"Number of samples : {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes : {len(target_names)}")

    print("\n========== FEATURE NAMES ==========")

    for i, name in enumerate(feature_names, start=1):
        print(f"{i:2}. {name}")

    print("\n========== TARGET CLASSES ==========")

    for i, name in enumerate(target_names):
        print(f"Class {i}: {name}")

    print("\n========== CLASS DISTRIBUTION ==========")

    class_counts = pd.Series(y).value_counts().sort_index()

    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

    print("=" * 60)


if __name__ == "__main__":
    X, y, feature_names, target_names = load_dataset()

    show_dataset_info(
        X,
        y,
        feature_names,
        target_names
    )