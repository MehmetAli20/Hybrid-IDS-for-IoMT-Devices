import os
import argparse
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

from data_loader import attack_type_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eğitilmiş modelle yeni bir CSV üzerinde tahmin")
    parser.add_argument("--run_dir", required=True, help="main.py'nin oluşturduğu outputs/<run> klasörü")
    parser.add_argument("--csv", required=True, help="CIC IoMT 2024 formatında özellik dosyası")
    args = parser.parse_args()

    artifacts = joblib.load(os.path.join(args.run_dir, 'preprocess.joblib'))
    model = keras.models.load_model(os.path.join(args.run_dir, 'model.h5'))

    df = pd.read_csv(args.csv)
    X = df[artifacts['features']].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
    X = artifacts['scaler'].transform(X).astype(np.float32)
    proba = model.predict(X, batch_size=4096, verbose=0)

    pred = proba.argmax(axis=1)
    groups, member_of = attack_type_groups(list(artifacts['label_encoder'].classes_))
    df['prediction'] = artifacts['label_encoder'].inverse_transform(pred)
    df['attack_type'] = np.array(groups)[np.array(member_of)[pred]]
    df['confidence'] = proba.max(axis=1)

    # Tahminler veri klasörüne değil run klasörüne yazılır (data/ altındaki CSV'ler eğitimde okunuyor)
    out = os.path.join(args.run_dir, os.path.basename(args.csv).replace('.csv', '_predictions.csv'))
    df.to_csv(out, index=False)
    print(df['prediction'].value_counts().to_string())
    print(f"\n✅ Tahminler yazıldı: {out}")
