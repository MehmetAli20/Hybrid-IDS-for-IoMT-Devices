import os
import json
import time
import argparse
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_and_preprocess_data
from evaluation import evaluate_predictions, plot_class_distribution, plot_feature_importances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest baseline: aynı veri hattı, aynı resmi test seti")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2)
    parser.add_argument("--max_per_class", type=int, default=50000)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--drop_features", nargs='*', default=['IAT'],
                        help="Çıkarılacak özellikler. IAT kayıt zamanı gibi davrandığı için varsayılan olarak "
                             "çıkarılır (bkz. IAT_SHORTCUT.md); tüm özellikler için değer vermeden --drop_features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    run_name = args.run_name or f"rf_{args.class_config}cls_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(project_dir, 'outputs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Ağaç modelleri ölçeklemeden etkilenmez; sınıf dengesi için örnekleme sınırı yeterli
    data = load_and_preprocess_data(
        os.path.join(project_dir, 'data'),
        args.class_config,
        max_per_class=args.max_per_class,
        balance='none',
        drop_features=args.drop_features,
        seed=args.seed
    )
    classes = list(data['label_encoder'].classes_)
    plot_class_distribution(data['class_counts'], classes, os.path.join(run_dir, 'class_distribution.png'))

    print(f"\n🌲 Random Forest eğitiliyor ({args.n_estimators} ağaç)...")
    rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed)
    rf.fit(data['X_train'], data['y_train'])

    proba = rf.predict_proba(data['X_test'])
    evaluate_predictions(data['y_test'], proba, classes, run_dir, data['test_subtypes'])

    importances = sorted(zip(data['features'], rf.feature_importances_), key=lambda x: -x[1])
    print("\n🔎 En önemli 10 özellik:")
    for name, value in importances[:10]:
        print(f"  • {name}: {value:.4f}")
    with open(os.path.join(run_dir, 'feature_importances.json'), 'w') as f:
        json.dump({name: float(value) for name, value in importances}, f, indent=2)
    plot_feature_importances(importances, os.path.join(run_dir, 'feature_importances.png'))
