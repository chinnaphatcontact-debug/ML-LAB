from tensorflow import keras
from tensorflow.keras import layers


# =========================
# Configuration
# =========================

INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 5


# =========================
# CNN-1
# =========================

def build_cnn_1():

    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),

            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Flatten(),

            layers.Dense(
                64,
                activation="relu"
            ),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="CNN_1_Simple"
    )

    return model


# =========================
# CNN-2
# =========================

def build_cnn_2():

    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),

            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Flatten(),

            layers.Dense(
                64,
                activation="relu"
            ),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="CNN_2_Medium"
    )

    return model


# =========================
# CNN-3
# =========================

def build_cnn_3():

    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),

            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same"
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Flatten(),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ],
        name="CNN_3_Deep"
    )

    return model


# =========================
# Test Models
# =========================

if __name__ == "__main__":

    models = [
        build_cnn_1(),
        build_cnn_2(),
        build_cnn_3()
    ]

    for model in models:

        print("\n" + "=" * 70)
        print(f"MODEL: {model.name}")
        print("=" * 70)

        model.summary()