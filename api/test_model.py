import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMAGE_SIZE = 256

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names
print("Class Names: ", class_names)

# Load the model
model = tf.keras.models.load_model(r"C:\Datasets\PlantVillage\npmodels\3.h5")

# Take a batch from the dataset
for images, labels in dataset.take(1):
    for i in range(5):  # Check first 5 images in the batch
        img = images[i].numpy().astype("uint8")
        actual_label = class_names[labels[i]]

        # Preprocess the image similarly
        img_resized = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        img_batch = np.expand_dims(img_resized, 0)

        # Predict using the model
        prediction = model.predict(img_batch)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        print(f"Actual label: {actual_label}, Predicted: {predicted_class}, Confidence: {confidence}")
        
        plt.imshow(img)
        plt.title(f"Actual: {actual_label}\nPredicted: {predicted_class}, Confidence: {confidence}")
        plt.axis("off")
        plt.show()
