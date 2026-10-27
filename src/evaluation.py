import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)


def _thin(x, y, n=2000):
    # Milyonlarca noktalı eğrileri çizmeden önce seyrelt
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def plot_confusion_matrix(cm, classes, path):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape), where=row_sums > 0)

    size = max(8, len(classes) * 0.6)
    plt.figure(figsize=(size + 2, size))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title('Normalize Edilmiş Confusion Matrix (resmi test seti)')
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)
    fontsize = 7 if len(classes) > 6 else 10
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', fontsize=fontsize,
                     color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_roc_pr(y_true, proba, classes, path):
    # İkili sınıflandırmada sadece 'Attack' sınıfı, çok sınıfta her sınıf için one-vs-rest
    targets = [classes.index('Attack')] if len(classes) == 2 else range(len(classes))

    # 10'dan fazla sınıfta varsayılan renkler tekrar ettiği için 20 renkli palet
    colors = plt.cm.tab20(np.arange(20)) if len(classes) > 10 else plt.cm.tab10(np.arange(10))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))
    for k, i in enumerate(targets):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        precision, recall, _ = precision_recall_curve(y_bin, proba[:, i])
        ap = average_precision_score(y_bin, proba[:, i])
        color = colors[k % len(colors)]
        ax_roc.plot(*_thin(fpr, tpr), color=color, label=f"{classes[i]} (AUC={auc(fpr, tpr):.3f})")
        ax_pr.plot(*_thin(recall, precision), color=color, label=f"{classes[i]} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC (one-vs-rest)')
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall (one-vs-rest)')
    ax_roc.legend(loc='lower right', fontsize=8)
    ax_pr.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history, path):
    h = history.history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(axes, ['loss', 'accuracy', 'macro_f1'], ['Loss', 'Accuracy', 'Macro-F1']):
        if key in h:
            ax.plot(h[key], label='Eğitim')
        if 'val_' + key in h:
            ax.plot(h['val_' + key], label='Doğrulama')
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_predictions(y_true, proba, classes, run_dir):
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(classes)))

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, digits=4,
                                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n📊 Resmi test seti sonuçları:")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print("\nSınıf bazlı metrikler:")
    print(report)

    metrics['per_class'] = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                                 output_dict=True, zero_division=0)
    with open(os.path.join(run_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, 'test_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    np.savetxt(os.path.join(run_dir, 'confusion_matrix.csv'), cm, fmt='%d', delimiter=',',
               header=','.join(classes), comments='')

    plot_confusion_matrix(cm, classes, os.path.join(run_dir, 'confusion_matrix.png'))
    plot_roc_pr(y_true, proba, classes, os.path.join(run_dir, 'roc_pr_curves.png'))
    return metrics
