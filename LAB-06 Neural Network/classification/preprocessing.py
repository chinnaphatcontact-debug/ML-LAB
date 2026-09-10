from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(X, y):
    """
    Split dataset into:
    - Training set      64%
    - Validation set    16%
    - Test set          20%

    Stratify is used to keep the class distribution balanced.
    """

    # First split: 80% training+validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Second split:
    # 80% of 80% = 64% training
    # 20% of 80% = 16% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_data(X_train, X_val, X_test):
    """
    Standardize input features using StandardScaler.

    The scaler is fitted ONLY on the training data.
    """

    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler for validation and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_data(X, y):
    """
    Complete preprocessing pipeline.
    """

    # Split dataset
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    # Standardize features
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = standardize_data(
        X_train,
        X_val,
        X_test
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler
    )


def show_preprocessing_info(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Display preprocessing results.
    """

    print("=" * 60)
    print("              PREPROCESSING RESULTS")
    print("=" * 60)

    print("\n========== DATA SPLIT ==========")

    print(f"Training set   : {X_train.shape[0]} samples")
    print(f"Validation set : {X_val.shape[0]} samples")
    print(f"Test set       : {X_test.shape[0]} samples")

    print(f"\nTotal samples  : {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")

    print("\n========== FEATURE SHAPE ==========")

    print(f"Training features   : {X_train.shape}")
    print(f"Validation features : {X_val.shape}")
    print(f"Test features       : {X_test.shape}")

    print("\n========== LABEL SHAPE ==========")

    print(f"Training labels   : {y_train.shape}")
    print(f"Validation labels : {y_val.shape}")
    print(f"Test labels       : {y_test.shape}")

    print("=" * 60)


if __name__ == "__main__":
    from data_loader import load_dataset

    # Load dataset
    X, y, feature_names, target_names = load_dataset()

    # Preprocess
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = preprocess_data(X, y)

    # Show results
    show_preprocessing_info(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\n========== STANDARDIZATION CHECK ==========")

    print(f"Training mean: {X_train.mean():.4f}")
    print(f"Training std : {X_train.std():.4f}")