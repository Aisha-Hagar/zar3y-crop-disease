import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

def compute_gradcam(model, img_array, intensity=0.5, res=224):
    base_model = model.get_layer("MobileNetV3Large")
    conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
            conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_predictions = grad_model(img_array)
        x = model.get_layer("global_average_pooling2d")(base_predictions)
        x = model.get_layer("dense")(x)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense_1")(x)
        
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    img = img_array[0]
    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    resized_heatmap = cv2.resize(heatmap, (res, res))
    color_heatmap = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1.0, color_heatmap, intensity, 0)
    return img, resized_heatmap, overlay

def save_side_by_side(original, overlay, save_path, title_text="Grad-CAM"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Input")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(title_text)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def get_gradcam_overlay_base64(model, img_array, intensity=0.5, res=224):
    """
    Returns base64-encoded PNG of Grad‑CAM overlay.
    model: Keras model (float)
    img_array: preprocessed image with shape (1, 224, 224, 3), values in [0,1]
    """
    _, _, overlay = compute_gradcam(model, img_array, intensity, res)
    # overlay is a numpy array (uint8) of shape (224,224,3)
    overlay_pil = Image.fromarray(overlay)
    buffered = BytesIO()
    overlay_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64