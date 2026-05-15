import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def compute_gradcam(model, img_array, intensity=0.5, res=224):
    """
    نسخة مخصصة للنماذج التي تحتوي على Sub-networks مثل MobileNetV3Large
    """
    # 1. الوصول للموديل الداخلي (الأساسي) والطبقة المطلوبة منه
    base_model = model.get_layer("MobileNetV3Large")
    
    # إيجاد آخر طبقة Convolution داخل الموديل الداخلي
    conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
            conv_layer_name = layer.name
            break

    # 2. بناء الـ Grad Model بناءً على الموديل الداخلي لمنع خطأ الـ ValueError
    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(conv_layer_name).output, base_model.output]
    )

    # 3. حساب التدرجات باستخدام الـ Tape
    with tf.GradientTape() as tape:
        # تمرير الصورة للموديل الداخلي مباشرة
        conv_outputs, base_predictions = grad_model(img_array)
        
        # نمرر مخرجات الموديل الداخلي لباقي طبقات الموديل الخارجي للحصول على التوقع النهائي الصحيح
        x = model.get_layer("global_average_pooling2d")(base_predictions)
        x = model.get_layer("dense")(x)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense_1")(x)
        
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # حساب التدرجات والـ Heatmap
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    # ReLU والتطبيع (Normalization) باستخدام NumPy
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    # تجهيز الصورة ودمج الـ Heatmap
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
