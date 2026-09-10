import tensorflow as tf


def build_model(
    input_dim=13,
    num_classes=3,
    hidden_layers=(16,)
):
    """
    Build a configurable Neural Network.

    Parameters:
        input_dim: Number of input features.
        num_classes: Number of output classes.
        hidden_layers: Tuple containing number of neurons
                       in each hidden layer.

    Returns:
        Compiled Keras model.
    """

    model = tf.keras.Sequential()

    # Input layer
    model.add(
        tf.keras.layers.Input(shape=(input_dim,))
    )

    # Hidden layers
    for neurons in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                neurons,
                activation="relu"
            )
        )

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_configurations():
    """
    Return Neural Network configurations
    for comparison.
    """

    configurations = {
        "NN_1_Hidden_16": (16,),
        "NN_2_Hidden_32_16": (32, 16),
        "NN_2_Hidden_64_32": (64, 32),
    }

    return configurations


def show_model_configurations():
    """
    Display the configurations used in this experiment.
    """

    configurations = get_configurations()

    print("=" * 60)
    print("          NEURAL NETWORK CONFIGURATIONS")
    print("=" * 60)

    for name, layers in configurations.items():
        print(f"{name}")
        print(f"  Hidden layers: {layers}")
        print()


if __name__ == "__main__":

    show_model_configurations()

    print("=" * 60)
    print("              TEST MODEL")
    print("=" * 60)

    model = build_model(
        input_dim=13,
        num_classes=3,
        hidden_layers=(16,)
    )

    model.summary()