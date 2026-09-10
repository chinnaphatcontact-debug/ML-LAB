import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data_loader import load_dataset
from preprocessing import preprocess_data


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "final_nn_model.keras"
)

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "graphs"
)


# Create graph directory if it does not exist
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("              EVALUATE FINAL MODEL")
    print("=" * 60)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\n========== LOAD MODEL ==========")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully")
    print(f"Model path: {MODEL_PATH}")


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("\n========== LOAD TEST DATA ==========")

    X, y, feature_names, target_names = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = preprocess_data(X, y)

    print(f"Test samples   : {X_test.shape[0]}")
    print(f"Test features  : {X_test.shape[1]}")


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    print("\n========== MODEL EVALUATION ==========")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


    # ========================================================
    # PREDICTION
    # ========================================================

    print("\n========== PREDICTIONS ==========")

    predictions = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(predictions, axis=1)

    prediction_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"Prediction Accuracy: {prediction_accuracy * 100:.2f}%")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)


    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix - Final Neural Network")

    plt.tight_layout()

    confusion_path = os.path.join(
        GRAPH_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()

    print(f"Confusion matrix saved to: {confusion_path}")


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )

    print(report)


    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print("\n========== SAMPLE PREDICTIONS ==========")

    print(
        f"{'Actual':<15}"
        f"{'Predicted':<15}"
        f"{'Confidence':<15}"
    )

    print("-" * 45)

    for i in range(len(y_test)):

        actual_class = target_names[y_test[i]]
        predicted_class = target_names[y_pred[i]]

        confidence = np.max(predictions[i]) * 100

        print(
            f"{actual_class:<15}"
            f"{predicted_class:<15}"
            f"{confidence:.2f}%"
        )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("              EVALUATION COMPLETE")
    print("=" * 60)