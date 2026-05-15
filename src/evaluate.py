import json
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def main():
    # Load metadata
    with open('data/class_names.json', 'r') as f:
        names = json.load(f)
    
    # Load best model
    model = tf.keras.models.load_model('models/best_model.keras')
    
    # Load test dataset (Preprocessing is baked into the model graph)
    ds = tf.keras.utils.image_dataset_from_directory(
        "data/plant_village/split/test", shuffle=False, image_size=(224,224), batch_size=32, class_names=names, label_mode='int', verbose=0
    )
    
    print("Evaluating model...")
    preds = model.predict(ds)
    y_p = np.argmax(preds, axis=1)
    y_t = np.concatenate([y.numpy() for x,y in ds], axis=0)
    
    # Generate report
    report_dict = classification_report(y_t, y_p, target_names=names, output_dict=True)
    accuracy = float(np.mean(y_t == y_p))
    macro_f1 = float(f1_score(y_t, y_p, average='macro'))
    
    final_report = {
        "overall_accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report_dict
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/eval_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Visualizations
    # 1. Confusion Matrix
    cm = confusion_matrix(y_t, y_p)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=names, yticklabels=names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc={accuracy:.4f}, F1={macro_f1:.4f})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    
    # 2. Training Curves
    if os.path.exists('outputs/training_history.json'):
        with open('outputs/training_history.json', 'r') as f:
            hist = json.load(f)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        all_acc = hist['phase_a']['accuracy'] + hist['phase_b']['accuracy']
        all_vacc = hist['phase_a']['val_accuracy'] + hist['phase_b']['val_accuracy']
        all_loss = hist['phase_a']['loss'] + hist['phase_b']['loss']
        all_vloss = hist['phase_a']['val_loss'] + hist['phase_b']['val_loss']
        
        axes[0].plot(all_acc, label='Train')
        axes[0].plot(all_vacc, label='Val')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        
        axes[1].plot(all_loss, label='Train')
        axes[1].plot(all_vloss, label='Val')
        axes[1].set_title('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/training_curves.png', dpi=150)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Evaluation Complete. Results saved in 'outputs/' folder.")

if __name__ == "__main__":
    main()
