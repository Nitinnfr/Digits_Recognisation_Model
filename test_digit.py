import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Image not found. Check the file path.")

    # Resize image to 28x28 pixels
    image = cv2.resize(image, (28, 28))

    # Invert image if background is white
    image = 255 - image

    # Normalize image
    image = image / 255.0

    # Reshape for model input
    image = image.reshape(1, 28, 28)

    return image


def predict_digit(image_path):
    model = tf.keras.models.load_model("mnist_digit_model.h5")

    image = preprocess_image(image_path)

    prediction = model.predict(image)
    digit = np.argmax(prediction)

    print(f"Predicted Digit: {digit}")

    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {digit}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = input("Enter handwritten digit image path: ")
    predict_digit(image_path)