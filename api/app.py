from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import uvicorn

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model_path = r'C:\Training\Trained.keras'
model = load_model(model_path)

# Define class names
CLASS_NAMES = ['Potato___Early_blight','Potato___Late_blight','Potato___healthy']

# Define a function to preprocess the image
def preprocess_image(file):
    img = image.load_img(file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image array
    return img_array

# Define a predict function
def predict(image_file):
    img = preprocess_image(image_file)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    return predicted_class, confidence, predictions

# Define an API endpoint to receive image files
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = io.BytesIO(contents)
        
        predicted_class, confidence, predictions = predict(img)
        class_name = CLASS_NAMES[predicted_class]
        
        return {"class_name": class_name, "confidence": float(confidence), "predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
