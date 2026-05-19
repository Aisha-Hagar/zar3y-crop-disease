import os
import io
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pathlib import Path
import sys
sys.path.append("src")  # allow import of Grad-CAM module
from grad_cam import get_gradcam_overlay_base64

app = FastAPI(title="Zar3y", description="Crop Disease Detection from Phone Photos")

# Global variables for the interpreter and class names
interpreter = None
class_names = None
keras_model = None

@app.on_event("startup")
async def load_model():
    global interpreter, class_names, keras_model
    # Load TFLite model
    BASE_DIR = Path(__file__).resolve().parent.parent
    #model_path = BASE_DIR / "models" / "model_quantized.tflite"
    #interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter = tf.lite.Interpreter(
        model_path="models/model_quantized.tflite",
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )

    interpreter.allocate_tensors()

    # Load Keras model (float) for Grad‑CAM
    keras_model = tf.keras.models.load_model("models/best_model.keras")

    # Load class names list (same order as during training)
    class_names = ["Corn_(maize)___Common_rust_", "Pepper,_bell___Bacterial_spot",
                   "Pepper,_bell___healthy", "Potato___Early_blight",
                   "Potato___Late_blight", "Potato___healthy",
                   "Tomato___Early_blight", "Tomato___Late_blight",
                   "Tomato___Leaf_Mold", "Tomato___healthy"]  # alphabetical order

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 2. Read and convert to PIL Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 3. Preprocess
    input_tensor = preprocess_image(image)

    # 4. Run TFLite inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    output_idx = output_details[0]['index']
    interpreter.set_tensor(input_idx, input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_idx)
    print("Input tensor min/max:", input_tensor.min(), input_tensor.max())
    print("Output logits:", output_data[0])

    # 5. Get prediction
    predicted_class_idx = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0]))
    predicted_class = class_names[predicted_class_idx]

    # 6. Generate Grad‑CAM overlay (using your existing function from Requirement 4)
    #    Note: generate_gradcam should take the original image (PIL) and return a base64 string
    gradcam_base64 = get_gradcam_overlay_base64(keras_model, input_tensor)

    # 7. Return JSON
    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "gradcam_base64": gradcam_base64
    })