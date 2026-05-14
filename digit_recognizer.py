import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def load_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()

    print("Training started...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    model.save("mnist_digit_model.h5")
    print("Model saved as mnist_digit_model.h5")

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_model()