import os
import json
import time
import argparse
import joblib
import tensorflow as tf

from data_loader import load_and_preprocess_data
from model import train_autoencoder, create_lstm_attention_model, train_model
from evaluation import evaluate_predictions, plot_class_distribution, plot_autoencoder_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder + BiLSTM + Attention ile IoMT saldırı tespiti")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Sınıf sayısı")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_per_class", type=int, default=50000,
                        help="Eğitimde sınıf başına en fazla örnek")
    parser.add_argument("--balance", choices=['weights', 'smote', 'none'], default='weights',
                        help="Sınıf dengesizliği: sınıf ağırlıkları, SMOTE ya da hiçbiri")
    parser.add_argument("--scaling", choices=['log', 'quantile', 'minmax'], default='log',
                        help="log: signed log1p + StandardScaler")
    parser.add_argument("--ae_mode", choices=['finetune', 'frozen', 'none'], default='finetune',
                        help="Autoencoder encoder'ı ince ayarlansın mı, dondurulsun mu, hiç kullanılmasın mı")
    parser.add_argument("--encoding_dim", type=int, default=32)
    parser.add_argument("--ae_epochs", type=int, default=30)
    parser.add_argument("--drop_features", nargs='*', default=['IAT'],
                        help="Çıkarılacak özellikler. IAT kayıt zamanı gibi davrandığı için varsayılan olarak "
                             "çıkarılır (bkz. IAT_SHORTCUT.md); tüm özellikler için değer vermeden --drop_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None,
                        help="Çıktı klasörü adı (varsayılan: sınıf sayısı + zaman)")
    args = parser.parse_args()

    # GPU bellek optimizasyonu
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU kullanılıyor." if gpus else "⚠️ GPU bulunamadı, CPU kullanılacak.")
    tf.keras.utils.set_random_seed(args.seed)

    project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    run_name = args.run_name or f"{args.class_config}cls_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(project_dir, 'outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    data = load_and_preprocess_data(
        os.path.join(project_dir, 'data'),
        args.class_config,
        max_per_class=args.max_per_class,
        balance=args.balance,
        scaling=args.scaling,
        drop_features=args.drop_features,
        seed=args.seed
    )
    classes = list(data['label_encoder'].classes_)
    plot_class_distribution(data['class_counts'], classes, os.path.join(run_dir, 'class_distribution.png'))

    encoder = None
    if args.ae_mode != 'none':
        encoder, ae_history = train_autoencoder(data['X_train'], data['X_val'], encoding_dim=args.encoding_dim,
                                                epochs=args.ae_epochs)
        plot_autoencoder_history(ae_history, os.path.join(run_dir, 'autoencoder_history.png'))
        encoder.trainable = args.ae_mode == 'finetune'

    model = create_lstm_attention_model(
        input_dim=data['X_train'].shape[1],
        num_classes=len(classes),
        encoder=encoder
    )
    model.summary()

    # Eğitim grafikleri ve training_log.csv her epoch sonunda run klasöründe güncellenir
    model, _ = train_model(
        model,
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        run_dir,
        classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=data['class_weights']
    )

    print("\n🧪 Resmi test setinde değerlendirme yapılıyor...")
    proba = model.predict(data['X_test'], batch_size=4096, verbose=0)
    evaluate_predictions(data['y_test'], proba, classes, run_dir, data['test_subtypes'])

    # Model ve ön işleme birlikte kaydedilir; predict.py ikisini birden kullanır
    model.save(os.path.join(run_dir, 'model.h5'))
    joblib.dump({
        'features': data['features'],
        'scaler': data['scaler'],
        'label_encoder': data['label_encoder'],
        'class_config': args.class_config
    }, os.path.join(run_dir, 'preprocess.joblib'))
    print(f"\n✅ Model ve çıktılar kaydedildi: {run_dir}")
