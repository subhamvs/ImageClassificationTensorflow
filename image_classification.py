import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob


class ImageClassification:
    def create_model_fashion(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Softmax(),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def load_data(self, data):
        (train_images, train_labels), (test_images, test_labels) = data.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        return train_images, train_labels, test_images, test_labels

    def train(self, model, train_images, train_labels, epochs, checkpoint_path):
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=1
        )
        model.fit(train_images, train_labels, epochs=epochs, callbacks=[callback])

    def evaluate(self, model, test_images, test_labels):
        predictions = model.predict(test_images)
        pred = [np.argmax(p) for p in predictions]
        return accuracy_score(pred, test_labels)

    def evaluate_for_all_models(
        self,
        test_images,
        test_labels,
        checkpoint_path,
        epochs,
        model_function,
        evaluate_function,
    ):
        model = model_function()
        for i in sorted(range(1, epochs + 1)):
            checkpoint = checkpoint_path + "/cp-%04d" % i + ".hdf5"
            print(checkpoint)
            model.load_weights(checkpoint)
            print(evaluate_function(model, test_images, test_labels))


# Epochs count
epochs = 5

# Create variables
fashion_mnist_data = tf.keras.datasets.fashion_mnist
fashion_mnist = ImageClassification()

# Create model and Data
model = fashion_mnist.create_model_fashion()
data = tf.keras.datasets.fashion_mnist

# Load Data
train_images, train_labels, test_images, test_labels = fashion_mnist.load_data(data)

# Checkpoint path
checkpoint_path = "/home/subha/Documents/Tensorflow-keras/beginner/image-classification/Training_1/cp-{epoch:04d}.hdf5"

# Train model
fashion_mnist.train(model, train_images, train_labels, epochs, checkpoint_path)

# Evaluate all models
fashion_mnist.evaluate_for_all_models(
    test_images,
    test_labels,
    "/home/subha/Documents/Tensorflow-keras/beginner/image-classification/Training_1/",
    epochs,
    fashion_mnist.create_model_fashion,
    fashion_mnist.evaluate,
)
