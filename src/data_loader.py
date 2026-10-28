import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# Dosya adındaki anahtar -> 19 sınıflı etiket
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

# 19 sınıflı etiket -> 6 sınıflı saldırı ailesi
ATTACK_CATEGORIES_6 = {
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}


def get_attack_category(file_name):
    for key in ATTACK_CATEGORIES_19:
        if key in file_name:
            return ATTACK_CATEGORIES_19[key]
    raise ValueError(f"Dosya adından etiket çıkarılamadı: {file_name}")


def map_labels(labels_19, class_config):
    labels_19 = np.asarray(labels_19, dtype=object)
    if class_config == 2:
        return np.where(labels_19 == 'Benign', 'Benign', 'Attack')
    if class_config == 6:
        return np.array([ATTACK_CATEGORIES_6[label] for label in labels_19], dtype=object)
    return labels_19


def attack_type_groups(classes):
    """Aynı saldırının DoS ve DDoS sürümlerini tek saldırı türünde toplar (DoS-UDP + DDoS-UDP -> DoS/DDoS-UDP).
    Pencere özellikleri saldırının kaç makineden geldiğini taşımadığı için bu ayrım ayrıca raporlanır.
    Tür adlarını ve her sınıfın tür indeksini döndürür."""
    keys = [c.replace('DDoS', 'DoS') for c in classes]
    names = {k: (k.replace('DoS', 'DoS/DDoS') if keys.count(k) > 1 else k) for k in keys}
    groups = list(dict.fromkeys(names[k] for k in keys))
    return groups, [groups.index(names[k]) for k in keys]


def load_split(data_dir, split):
    """data/<split> altındaki tüm CSV'leri okur. İlk okumadan sonra data/cache altına yazar;
    CSV'ler değişirse data/cache klasörünü silin."""
    cache_path = os.path.join(data_dir, 'cache', f'{split}.pkl')
    if os.path.exists(cache_path):
        print(f"  ✓ {split} önbellekten okundu: {cache_path}")
        return pd.read_pickle(cache_path)

    folder = os.path.join(data_dir, split)
    dfs = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, fn))
            df = df.replace([np.inf, -np.inf], np.nan).astype(np.float32)
            df['label'] = get_attack_category(fn)
            dfs.append(df)
            print(f"  ✓ {fn}: {len(df)} kayıt")
    data = pd.concat(dfs, ignore_index=True)
    data['label'] = data['label'].astype('category')

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data.to_pickle(cache_path)
    return data


def cap_per_class(X, y, max_per_class, seed=42):
    """Her sınıftan en fazla max_per_class örneği rastgele bırakır."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = rng.permutation(np.concatenate(keep))
    return X[keep], y[keep]


def signed_log1p(X):
    # Rate, Header_Length, IAT gibi çok çarpık özellikleri sıkıştırır
    return np.sign(X) * np.log1p(np.abs(X))


def build_scaler(scaling):
    if scaling == 'log':
        return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler())
    if scaling == 'quantile':
        return QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=200000, random_state=42)
    return MinMaxScaler()


def print_distribution(title, y, classes):
    counts = np.bincount(y, minlength=len(classes))
    print(f"\n📊 {title}:")
    for name, count in zip(classes, counts):
        print(f"  • {name}: {count}")


def load_and_preprocess_data(
    data_dir,
    class_config=2,
    max_per_class=50000,
    balance='weights',
    scaling='log',
    drop_features=(),
    val_size=0.15,
    seed=42
):
    # 1) Oku ve etiketle
    print("📁 Eğitim verisi okunuyor...")
    train = load_split(data_dir, 'train')
    print("📁 Test verisi okunuyor (resmi test seti)...")
    test = load_split(data_dir, 'test')

    features = [c for c in train.columns if c != 'label' and c not in drop_features]
    if drop_features:
        print(f"  • Çıkarılan özellikler: {list(drop_features)}")

    # 2) Eğitimdeki birebir tekrarları at (dosyalar arası dahil); test resmi haliyle kalır
    before = len(train)
    train = train.drop_duplicates(ignore_index=True)
    print(f"  • Eğitimde tekrar eden {before - len(train)} satır atıldı, kalan {len(train)}")

    X = train[features].fillna(0).values
    y = map_labels(train['label'], class_config)
    X_test = test[features].fillna(0).values
    y_test = map_labels(test['label'], class_config)
    del train, test

    # 3) Doğrulama setini eğitimden ayır; resmi test setine sadece en sonda dokunulur
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y)
    del X, y

    # 4) Büyük sınıfları sınırla (doğrulamada aynı oranda)
    X_train, y_train = cap_per_class(X_train, y_train, max_per_class, seed)
    X_val, y_val = cap_per_class(X_val, y_val, int(max_per_class * val_size / (1 - val_size)), seed)

    # 5) Ölçekleme: istatistikler sadece eğitim setinden öğrenilir
    scaler = build_scaler(scaling)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 6) Etiketleri sayıya çevir
    le = LabelEncoder().fit(y_train)
    y_train, y_val, y_test = le.transform(y_train), le.transform(y_val), le.transform(y_test)
    classes = list(le.classes_)
    print_distribution("Eğitim seti", y_train, classes)

    # 7) Sınıf dengesizliği: SMOTE (azınlıkları max_per_class'a tamamlar) veya sınıf ağırlıkları
    class_weights = None
    if balance == 'smote':
        counts = np.bincount(y_train)
        strategy = {i: max_per_class for i, c in enumerate(counts) if c < max_per_class}
        if strategy:
            X_train, y_train = SMOTE(sampling_strategy=strategy, random_state=seed).fit_resample(X_train, y_train)
            print_distribution("SMOTE sonrası eğitim seti", y_train, classes)
    elif balance == 'weights':
        counts = np.bincount(y_train)
        class_weights = {i: float(np.sqrt(counts.max() / c)) for i, c in enumerate(counts)}

    print(f"\n✅ Eğitim {len(y_train)} | Doğrulama {len(y_val)} | Test {len(y_test)} örnek")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': le,
        'scaler': scaler,
        'features': features,
        'class_weights': class_weights,
    }
