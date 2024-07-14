# main.py

from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load your trained TensorFlow model
model = tf.keras.models.load_model("./PlantVillage/npmodels/3")  # Update with your model path

# Define prediction function
def predict(image):
    # Preprocess the image (resize, normalize, etc.)
    img_array = tf.image.resize(image, [256, 256])  # Assuming IMAGE_SIZE is 256x256
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

# Route to handle image upload and prediction
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.image.decode_image(contents, channels=3)
    prediction = predict(image)
    return {"class_id": prediction}

# Main function to run the app with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
