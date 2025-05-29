import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import gc

from data_loader import load_and_preprocess_data
from model import create_lstm_attention_model, train_model

# Başlangıç testi
print("🟢 main.py icinde")

def evaluate_model(model, X_test, y_test, label_encoder):
    print("\n🧪 Model degerlendirmesi basliyor...")

    # Test verilerini batches halinde değerlendir
    batch_size = 128
    n_batches = (len(X_test) + batch_size - 1) // batch_size

    test_loss = 0
    test_accuracy = 0
    all_predictions = []
    all_true_labels = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))

        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]

        # Batch değerlendirmesi
        batch_metrics = model.evaluate(X_batch, y_batch, verbose=0)
        batch_loss = batch_metrics[0]
        batch_accuracy = batch_metrics[1]

        test_loss += batch_loss * (end_idx - start_idx) / len(X_test)
        test_accuracy += batch_accuracy * (end_idx - start_idx) / len(X_test)

        # Tahminler
        batch_preds = model.predict(X_batch, verbose=0)
        all_predictions.append(batch_preds)
        all_true_labels.append(y_batch)

    # Sonuçları birleştir
    y_pred_cat = np.vstack(all_predictions)
    y_true_cat = np.vstack(all_true_labels)

    # Metrikleri hesapla
    y_pred_encoded = y_pred_cat.argmax(axis=1)
    y_true_encoded = y_true_cat.argmax(axis=1)

    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_true = label_encoder.inverse_transform(y_true_encoded)

    # Detaylı metrikler
    print("\n📊 Detaylı Değerlendirme Sonuçları:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Sınıf bazlı metrikler
    print("\nSınıf Bazlı Metrikler:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Normalize edilmiş Confusion Matrix
    row_sums = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            cm_normalized[i] = cm[i] / row_sums[i]

    # Confusion Matrix görselleştirme
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalize Edilmiş Confusion Matrix')
    plt.colorbar()

    classes = label_encoder.classes_
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC eğrisi ve AUC (ikili sınıflandırma için)
    if len(label_encoder.classes_) == 2:
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

        fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_cat[:, 1])
        auc = roc_auc_score(y_true_encoded, y_pred_cat[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

        # Precision-Recall eğrisi (binary)
        precision, recall, _ = precision_recall_curve(y_true_encoded, y_pred_cat[:, 1])
        average_precision = average_precision_score(y_true_encoded, y_pred_cat[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig('pr_curve.png')
        plt.close()
    else:
        # Multi-class için PR ve ROC eğrileri one-vs-rest olarak çizilir
        from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

        n_classes = y_true_cat.shape[1]
        for i, class_name in enumerate(label_encoder.classes_):
            # PR Curve (One-vs-Rest)
            precision, recall, _ = precision_recall_curve(y_true_cat[:, i], y_pred_cat[:, i])
            average_precision = average_precision_score(y_true_cat[:, i], y_pred_cat[:, i])

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve: {class_name}')
            plt.legend(loc="lower left")
            plt.savefig(f'pr_curve_{class_name}.png')
            plt.close()

            # ROC Curve (One-vs-Rest)
            fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_cat[:, i])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {class_name}')
            plt.legend(loc="lower right")
            plt.savefig(f'roc_curve_{class_name}.png')
            plt.close()

    return test_loss, test_accuracy, y_pred, y_true


# Ana çalışma bloğu
if __name__ == "__main__":
    print("✅ __main__ bloğuna girildi.")

    # GPU bellek optimizasyonu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU bellek büyümesi etkinleştirildi.")
        except RuntimeError as e:
            print(f"⚠️ GPU bellek ayarı hatası: {e}")

    parser = argparse.ArgumentParser(description="Autoencoder + LSTM model for intrusion detection")
    
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classification categories")
    parser.add_argument("--max_samples", type=int, default=100000, 
                        help="Maximum samples per class")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    print("📁 Veri yükleniyor...")

    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, label_encoder = load_and_preprocess_data(
        data_dir, 
        args.class_config,
        max_samples=args.max_samples,
        encoding_dim=32   # veya istediğin encoding boyutu
    )

    print("✅ Veri yüklendi.")

    print("📊 Sınıf dağılımı:")
    unique, counts = np.unique(y_train_cat.argmax(axis=1), return_counts=True)
    for class_idx, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([class_idx])[0]
        print(f"{class_name}: {count}")

    # Bellek temizliği
    gc.collect()

    model = create_lstm_attention_model(
    input_dim=X_train.shape[1],
    num_classes=y_train_cat.shape[1],
    reshape_dim=8  # veya AE encoding_dim'e göre uygun bir çarpan seç (ör. 16, 8, 4)
    )

    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU kullanılabilir durumda.")
    else:
        print("⚠️ GPU bulunamadı, CPU kullanılacak.")

    print("🚀 Model eğitiliyor...")
    model, history = train_model(
        model, 
        X_train, 
        y_train_cat, 
        X_val, 
        y_val_cat, 
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Gereksiz değişkenleri temizle
    gc.collect()

    # Eğitim grafikleri
    plt.figure(figsize=(15, 10))
    
    # Accuracy grafiği
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss grafiği
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Precision grafiği
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Recall grafiği
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Model değerlendirmesi
    test_loss, test_accuracy, y_pred, y_true = evaluate_model(model, X_test, y_test_cat, label_encoder)
    
    # Model kaydetme
    model.save('final_model.h5', save_format='h5')
    print("✅ Model başarıyla kaydedildi: final_model.h5")

    print(f"TensorFlow version: {tf.__version__}")
