import os
import json
import time
import argparse
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_split, load_and_preprocess_data
from evaluation import evaluate_predictions


def iat_by_class(data_dir):
    # 19 sınıfın her biri için IAT dağılımı; eğitim tekrarları ana veri hattındaki gibi atılır
    stats = {}
    for split in ['train', 'test']:
        df = load_split(data_dir, split)
        if split == 'train':
            df = df.drop_duplicates(ignore_index=True)
        s = df.groupby('label', observed=True)['IAT'].describe(percentiles=[0.05, 0.5, 0.95])
        s = s[['5%', '50%', '95%']]
        s.columns = [f'{split}_p05', f'{split}_median', f'{split}_p95']
        stats[split] = s
        del df

    table = stats['train'].join(stats['test'])
    table.index = table.index.astype(str)
    return table.sort_values('train_median')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAT kısayol analizi: sınıf bazında IAT dağılımı ve Random Forest "
                                                 "ile tüm özellikler / sadece IAT / IAT'siz karşılaştırması")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=6)
    parser.add_argument("--max_per_class", type=int, default=50000)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data_dir = os.path.join(project_dir, 'data')
    run_name = args.run_name or f"iat_{args.class_config}cls_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(project_dir, 'outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("⏱️ Sınıf bazında IAT dağılımı hesaplanıyor...")
    table = iat_by_class(data_dir)
    table.to_csv(os.path.join(run_dir, 'iat_by_class.csv'), float_format='%.1f')
    print(table.to_string(float_format=lambda v: f"{v:,.0f}"))

    # Üç deney aynı satırları kullanır, sadece sütunlar değişir; ağaç modelleri ölçeklemeden etkilenmez
    data = load_and_preprocess_data(data_dir, args.class_config, max_per_class=args.max_per_class,
                                    balance='none', drop_features=[], seed=args.seed)
    classes = list(data['label_encoder'].classes_)
    features = data['features']
    iat = features.index('IAT')
    variants = {
        'all_features': list(range(len(features))),
        'IAT_only': [iat],
        'no_IAT': [i for i in range(len(features)) if i != iat],
    }

    summary = {}
    for name, cols in variants.items():
        print(f"\n🌲 Random Forest: {name} ({len(cols)} özellik)")
        rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed)
        rf.fit(data['X_train'][:, cols], data['y_train'])
        proba = rf.predict_proba(data['X_test'][:, cols])

        variant_dir = os.path.join(run_dir, name)
        os.makedirs(variant_dir, exist_ok=True)
        metrics = evaluate_predictions(data['y_test'], proba, classes, variant_dir)
        summary[name] = {'n_features': len(cols), 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']}
        if name == 'all_features':
            ranked = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])
            summary[name]['top_features'] = [(f, float(v)) for f, v in ranked[:5]]

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n📋 Özet (resmi test seti):")
    for name, s in summary.items():
        print(f"  • {name:12s} {s['n_features']:2d} özellik | Accuracy {s['accuracy']:.4f} | Macro-F1 {s['macro_f1']:.4f}")
    print(f"\n✅ Çıktılar kaydedildi: {run_dir}")
