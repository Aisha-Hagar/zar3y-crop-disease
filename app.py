import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Zar3y", layout="wide")
st.title("Zar3y — Crop Disease Detection from Phone Photos")
st.markdown("Upload a photo of a tomato, potato, pepper, or corn leaf to get a diagnosis.")

def get_explanation(class_name):
    explanations = {
        "Corn_(maize)___Common_rust_": {
            "description": "Common corn (maize) rust is a fungal disease caused by the pathogen Puccinia sorghi. It is characterized by small, powdery, brick-red to dark brown pustules that appear on both the upper and lower surfaces of leaves.",
            "next_steps": "Planting resistant hybrids, which limits spore production. For active infections, foliar fungicides can be applied."
        },
        "Pepper,_bell___Bacterial_spot": {
            "description": "Bacterial spot in bell peppers (caused by Xanthomonas species) is a highly destructive disease that thrives in warm, wet conditions. It affects all above-ground parts of the plant, causing water-soaked leaf lesions that turn dark, leaf yellowing and drop, and raised, scab-like spots on the fruit.",
            "next_steps": " Immediately apply a copper-based fungicide or Serenade bio-fungicide to existing foliage. For severe infections, remove affected plants entirely."
        },
        "Pepper,_bell___healthy": {
            "description": "Healthy pepper plant leaves are defined by their deep green color, smooth edges, and distinct lanceolate-to-ovate shape (averaging \(10\) to \(20\text{ cm}\) long). They feel firm, feature a glossy upper surface, and are slightly lighter on the matte underside.",
            "next_steps": "For pepper leaves to stay healthy, you must address the specific symptoms (like curling, yellowing, or spots) using targeted care. Common treatments include adjusting your watering schedule, applying gentle foliar sprays like Epsom Salt, or controlling pests with neem oil."
        },
        "Potato___Early_blight": {
            "description": "Potato early blight is a common fungal disease caused by Alternaria solani, affecting stressed or senescing plants. It primarily attacks foliage but can also affect stems and tubers, leading to reduced crop yields and defoliation if left unmanaged.",
            "next_steps": " Promptly remove infected leaves and apply a copper-based or chemical protectant fungicide. For severe infections, alternate between systemic fungicides, cut off affected stems to prevent the fungus from reaching the tubers, and maintain optimum soil nutrition and airflow."
        },
        "Potato___Late_blight": {
            "description": "Late blight is a devastating, fast-spreading plant disease caused by the oomycete Phytophthora infestans. It primarily attacks potatoes and tomatoes, thriving in cool, moist, and humid weather conditions. The disease is infamous for destroying entire crops within days if left unmanaged.",
            "next_steps": "There is no chemical cure for already infected foliage. Treatment relies entirely on stopping the spread with targeted fungicides and implementing strict cultural practices to protect your healthy tubers."
        },
        "Potato___healthy": {
            "description": "A healthy potato leaf is a compound, spirally arranged leaf measuring 20–30 cm long. It features a distinct structure composed of one large terminal leaflet and two to four pairs of smaller, oval leaflets with pointed tips and smooth or wavy margins.",
            "next_steps": "Ensure proper spacing for airflow and avoid overhead watering."
        },
        "Tomato___Early_blight": {
            "description": "Early blight is a widespread fungal disease of tomatoes caused by Alternaria solani or Alternaria tomatophila. It thrives in warm, humid weather and primarily attacks older, lower leaves, causing dark brown spots with concentric rings.",
            "next_steps": "Immediately prune and dispose of infected, yellowed leaves to stop the fungus from spreading. Follow up by applying a protective fungicide (either chemical or organic) to the foliage, and implement preventative cultural practices like crop rotation and bottom-up watering."
        },
        "Tomato___Late_blight": {
            "description": "Late blight appears as dark, water‑soaked lesions on leaves and stems. It spreads quickly in cool, wet conditions.",
            "next_steps": "Apply a copper‑based fungicide immediately. Remove and destroy infected leaves. Avoid overhead watering."
        },
        "Tomato___Leaf_Mold": {
        "description": "Tomato leaf mold is a fungal disease caused by Passalora fulva (formerly Fulvia fulva) that primarily targets tomato foliage in high-humidity environments like greenhouses. It is identified by pale green/yellow spots on the upper leaf surface and a distinct, velvety olive-green mold on the underside.",
        "next_steps": "immediately pruning affected foliage and destroying it. Improve air circulation through wide plant spacing and trimming lower leaves, and lower ambient humidity. Apply organic treatments like wettable sulfur, copper-based fungicides, or a mild baking soda-oil-soap spray."
    },
        "Tomato___healthy": {
        "description": "A healthy tomato leaf is vibrant to dark green, matte-textured, and highly aromatic when crushed. It features a unique compound structure composed of multiple smaller leaflets attached to a central stem.",
        "next_steps": "Start by ensuring good airflow, keeping the soil moisture consistent, and plucking off any discolored leaves immediately. Treat minor fungal issues with natural sprays and use organic pest control to tackle insects that damage foliage."
    }
    }
    return explanations.get(class_name, {
        "description": "No specific description available for this class.",
        "next_steps": "Contact your local agricultural extension agent for advice."
    })

input_method = st.radio("Choose input method:", ["Upload image file", "Take a photo with camera"])

image = None
if input_method == "Upload image file":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif input_method == "Take a photo with camera":
    camera_photo = st.camera_input("Capture a photo")
    if camera_photo is not None:
        image = Image.open(camera_photo)

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    if st.button("Diagnose"):
        with st.spinner("Analyzing..."):
            # Convert PIL image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # Call FastAPI backend
            files = {"file": ("leaf.jpg", img_bytes, "image/jpeg")}
            try:
                response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.write(f"**Class:** {result['predicted_class']}")
            st.write(f"**Confidence:** {result['confidence']:.2%}")

            # Static plain‑language explanation and next‑step suggestions
            explanation = get_explanation(result['predicted_class'])
            st.subheader("What does this mean?")
            st.write(explanation["description"])
            st.subheader("Next Steps")
            st.write(explanation["next_steps"])

        with col2:
            st.subheader("Model Attention (Grad‑CAM)")
            # Decode base64 image
            gradcam_bytes = base64.b64decode(result['gradcam_base64'])
            gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
            st.image(gradcam_image, use_column_width=True)