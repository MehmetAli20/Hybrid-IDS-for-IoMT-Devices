import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, PercentFormatter
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)

from data_loader import attack_type_groups

# Renk körlüğüne karşı doğrulanmış kategorik palet (sıra sabit, en fazla 8 seri) ve sade eksen tonları
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUES = LinearSegmentedColormap.from_list('blues', [SURFACE, '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'text.color': INK,
    'axes.titlelocation': 'left', 'axes.titlesize': 11, 'axes.prop_cycle': cycler(color=SERIES),
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1, 'grid.linestyle': '-',
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'legend.labelcolor': INK_2,
})


def _save(fig, path, title=None):
    # Figür başlığı varsa alt grafiklerin başlıklarıyla çakışmasın diye üstte yer bırakılır
    top = 1
    if title:
        fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=12)
        top = 1 - 0.35 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _compact(value, _pos=None):
    for div, suffix in ((1e6, 'M'), (1e3, 'K')):
        if abs(value) >= div:
            return f"{value / div:g}{suffix}"
    return f"{value:g}"


def _grid_shape(n):
    # Küçük çoklu grafikler için satır x sütun
    ncols = 2 if n <= 2 else 3 if n <= 9 else 5
    return int(np.ceil(n / ncols)), ncols


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
    fig, ax = plt.subplots(figsize=(size + 2, size))
    image = ax.imshow(cm_norm, interpolation='nearest', cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    fig.colorbar(image, ax=ax)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes)
    fontsize = 7 if len(classes) > 6 else 10
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', fontsize=fontsize,
                    color='white' if cm_norm[i, j] > 0.5 else INK)

    ax.set(xlabel='Tahmin edilen sınıf', ylabel='Gerçek sınıf',
           title='Normalize edilmiş confusion matrix (resmi test seti)')
    _save(fig, path)


def plot_roc_pr(y_true, proba, classes, path):
    # İkili sınıflandırmada sadece 'Attack' sınıfı, çok sınıfta her sınıf için one-vs-rest
    targets = [classes.index('Attack')] if len(classes) == 2 else range(len(classes))
    curves = []
    for i in targets:
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        precision, recall, _ = precision_recall_curve(y_bin, proba[:, i])
        curves.append((classes[i], _thin(fpr, tpr), auc(fpr, tpr), _thin(recall, precision),
                       average_precision_score(y_bin, proba[:, i])))

    if len(curves) <= len(SERIES):
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6.5))
        for (name, roc_xy, roc_auc, pr_xy, ap), color in zip(curves, SERIES):
            ax_roc.plot(*roc_xy, color=color, label=f"{name} (AUC {roc_auc:.3f})")
            ax_pr.plot(*pr_xy, color=color, label=f"{name} (AP {ap:.3f})")
        # Tek ortak açıklama grafiklerin altında; eğrilerin üstüne binmez
        fig.legend(ax_roc.get_lines(), [f"{name} (AUC {roc_auc:.3f} · AP {ap:.3f})"
                                        for name, _, roc_auc, _, ap in curves],
                   loc='upper center', bbox_to_anchor=(0.5, 0), ncol=min(3, len(curves)), fontsize=9)
        ax_roc.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        ax_roc.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC (one-vs-rest)')
        ax_pr.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall (one-vs-rest)')
        _save(fig, path)
        return

    # 8'den fazla sınıfta renkler ayırt edilemez; her sınıf kendi küçük precision-recall grafiğinde
    nrows, ncols = _grid_shape(len(curves))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (name, _, roc_auc, pr_xy, ap) in zip(axes.flat, curves):
        ax.plot(*pr_xy, color=SERIES[0])
        ax.set_title(f"{name}\nAP {ap:.3f} · ROC AUC {roc_auc:.3f}", fontsize=9)
    for ax in axes.flat[len(curves):]:
        ax.set_visible(False)
    _save(fig, path, title='Precision-recall, sınıf başına (x: recall, y: precision)')


def plot_class_distribution(counts, classes, path):
    """Sınıf başına satır sayısını çizer ve aynı tabloyu yanına CSV olarak yazar.
    Sayılar yüzlerden milyonlara uzandığı için log ölçekli nokta grafik kullanılır."""
    pd.DataFrame(counts, index=classes).to_csv(path.replace('.png', '.csv'), index_label='class')

    fig, ax = plt.subplots(figsize=(9, max(2.6, 0.42 * len(classes) + 1.4)))
    rows = np.arange(len(classes))
    offsets = np.linspace(-0.2, 0.2, len(counts))
    for (name, values), offset, color in zip(counts.items(), offsets, SERIES):
        ax.plot(values, rows + offset, 'o', color=color, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(_compact))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(rows)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set(xlabel='Satır sayısı (log ölçek)', title='Sınıf dağılımı')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    _save(fig, path)


def plot_autoencoder_history(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], color=SERIES[0], label='Eğitim')
    ax.plot(epochs, history['val_loss'], color=SERIES[1], label='Doğrulama')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel='Epoch', ylabel='MSE (log ölçek)', title='Autoencoder yeniden oluşturma kaybı')
    ax.legend()
    _save(fig, path)


def plot_training_curves(log, classes, run_dir):
    """Eğitim sürerken her epoch sonunda yeniden çizilir: kayıp, accuracy, doğrulama macro-F1'i,
    öğrenme oranı ve sınıf başına doğrulama F1'i."""
    epochs = log['epoch']
    f1 = log['val_macro_f1']
    best = int(f1.idxmax())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, key, title in zip(axes[:2], ['loss', 'accuracy'], ['Kayıp', 'Accuracy']):
        ax.plot(epochs, log[key], color=SERIES[0], label='Eğitim')
        ax.plot(epochs, log['val_' + key], color=SERIES[1], label='Doğrulama')
        ax.set_title(title)
        ax.legend()
    ax = axes[2]
    ax.plot(epochs, f1, color=SERIES[1])
    ax.plot(epochs[best], f1[best], 'o', color=SERIES[1], markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ax.set_title(f"Doğrulama macro-F1 (en iyi {f1[best]:.4f}, epoch {epochs[best]})")
    ax = axes[3]
    ax.plot(epochs, log['lr'], color=INK_2)
    ax.set_yscale('log')
    ax.set_title('Öğrenme oranı')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _save(fig, os.path.join(run_dir, 'training_history.png'))

    # Her sınıf kendi küçük grafiğinde; dikey çizgi en iyi epoch
    nrows, ncols = _grid_shape(len(classes))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, name in zip(axes.flat, classes):
        values = log[f'val_f1_{name}']
        ax.plot(epochs, values, color=SERIES[0])
        ax.axvline(epochs[best], color=AXIS, linewidth=1)
        ax.set_title(f"{name}: {values[best]:.2f}", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    _save(fig, os.path.join(run_dir, 'val_f1_per_class.png'),
          title='Sınıf başına doğrulama F1 (dikey çizgi ve değer: en iyi epoch)')


def plot_per_class_metrics(report, classes, path, attack_type=None):
    panels = 2 if attack_type else 1
    fig, axes = plt.subplots(1, panels, figsize=(8 * panels, max(2.6, 0.5 * len(classes) + 1.4)), squeeze=False)

    ax = axes[0, 0]
    rows = np.arange(len(classes))
    for k, (key, label) in enumerate([('precision', 'Precision'), ('recall', 'Recall'), ('f1-score', 'F1')]):
        ax.barh(rows + (k - 1) * 0.26, [report[c][key] for c in classes], height=0.22, color=SERIES[k],
                label=label)
    for i, name in enumerate(classes):
        value = report[name]['f1-score']
        ax.text(value + 0.01, i + 0.26, f"{value:.2f}", va='center', fontsize=8, color=INK_2)
    ax.set_yticks(rows)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.grid(axis='y', visible=False)
    ax.set_title('Sınıf bazında test metrikleri (değer: F1)', pad=26)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=3, fontsize=9)

    if attack_type:
        ax = axes[0, 1]
        names = attack_type['groups']
        values = [attack_type['per_type_f1'][name] for name in names]
        rows = np.arange(len(names))
        ax.barh(rows, values, height=0.25, color=SERIES[2])
        for i, value in enumerate(values):
            ax.text(value + 0.01, i, f"{value:.2f}", va='center', fontsize=8, color=INK_2)
        ax.set_yticks(rows)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.grid(axis='y', visible=False)
        ax.set_title(f"Saldırı türü seviyesinde F1 (DoS/DDoS ayrımı: {attack_type['dos_vs_ddos_accuracy']:.3f})",
                     pad=26)
    _save(fig, path)


def plot_confidence(y_true, proba, path):
    # Model ne kadar emin: doğru ve yanlış tahminlerde en yüksek olasılığın dağılımı (her grup kendi içinde oran)
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    for mask, label, color in [(correct, 'Doğru', SERIES[0]), (~correct, 'Yanlış', SERIES[1])]:
        if mask.any():
            ax.hist(confidence[mask], bins=bins, weights=np.full(mask.sum(), 1 / mask.sum()), histtype='step',
                    linewidth=2, color=color,
                    label=f"{label}: {mask.sum():,} pencere, ortalama güven {confidence[mask].mean():.2f}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel='Güven (en yüksek sınıf olasılığı)', ylabel='Grubun içindeki pay')
    ax.set_title('Tahmin güveni: doğru ve yanlış tahminler', pad=40)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), fontsize=9)
    _save(fig, path)


def plot_detection_by_subtype(subtypes, flagged, path):
    """Her saldırı alt türünün (19 sınıflı etiket) ne kadarının saldırı olarak işaretlendiğini çizer; Benign'de aynı
    oran yanlış alarm oranıdır. Toplam isabetin gizlediği kör noktaları gösterir. Tablo CSV olarak da yazılır."""
    table = pd.DataFrame({'subtype': subtypes, 'flagged': flagged}).groupby('subtype')['flagged'].agg(['mean', 'size'])
    table.columns = ['flagged_rate', 'windows']
    table.to_csv(path.replace('.png', '.csv'))

    attacks = table.drop(index='Benign').sort_values('flagged_rate')
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(attacks) + 1.8))
    rows = np.arange(len(attacks))
    ax.barh(rows, attacks['flagged_rate'], height=0.6, color=SERIES[0])
    for i, rate in enumerate(attacks['flagged_rate']):
        ax.text(rate + 0.01, i, f"{rate:.1%}", va='center', fontsize=8, color=INK_2)
    ax.set_yticks(rows)
    ax.set_yticklabels(attacks.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis='y', visible=False)
    ax.set(xlabel='Saldırı olarak işaretlenen pencere oranı',
           title=f"Saldırı türüne göre tespit oranı (Benign'de yanlış alarm: {table.loc['Benign', 'flagged_rate']:.1%})")
    _save(fig, path)
    return table


def plot_feature_importances(importances, path, top=20):
    names = [name for name, _ in importances[:top]]
    values = [value for _, value in importances[:top]]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.3))
    rows = np.arange(len(names))
    ax.barh(rows, values, height=0.6, color=SERIES[0])
    ax.text(values[0], 0, f"  {values[0]:.3f}", va='center', fontsize=9, color=INK_2)
    ax.set_yticks(rows)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set(xlabel='Önem (ortalama saflık kazancı)', title=f'Random Forest: en önemli {len(names)} özellik')
    _save(fig, path)


def plot_iat_by_class(table, path):
    # Sadece flood sınıfları (bantları 84 milyon civarında); diğerlerinde IAT 0 ile 169 milyon arasında ikiye bölünüyor
    floods = table[table['train_p05'] > 1e6]
    if floods.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(floods) + 1.5))
    rows = np.arange(len(floods))
    for split, offset, color, label in [('train', -0.15, SERIES[0], 'Eğitim'), ('test', 0.15, SERIES[1], 'Test')]:
        ax.hlines(rows + offset, floods[f'{split}_p05'], floods[f'{split}_p95'], colors=color, linewidth=2)
        ax.plot(floods[f'{split}_median'], rows + offset, 'o', color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_yticks(rows)
    ax.set_yticklabels(floods.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.grid(axis='y', visible=False)
    ax.set(xlabel='IAT', title='Flood sınıflarında IAT (nokta: medyan, çizgi: %5–%95)')
    ax.legend(loc='upper right')
    _save(fig, path)


def attack_type_metrics(cm, classes):
    """Karışıklık matrisini saldırı türü seviyesine indirger; DoS/DDoS çifti yoksa (2 sınıf) None döner."""
    groups, member_of = attack_type_groups(classes)
    paired = [g for g in range(len(groups)) if member_of.count(g) > 1]
    if not paired:
        return None

    onehot = np.eye(len(groups), dtype=np.int64)[member_of]
    cm_type = onehot.T @ cm @ onehot
    tp = np.diag(cm_type).astype(float)
    predicted, actual = cm_type.sum(axis=0), cm_type.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)

    # Türü doğru bilinen DoS/DDoS pencerelerinde DoS mu DDoS mu sorusunun isabeti
    exact = sum(cm[i, i] for i, g in enumerate(member_of) if g in paired)
    same_type = sum(cm_type[g, g] for g in paired)
    return {
        'groups': groups,
        'accuracy': float(tp.sum() / cm_type.sum()),
        'macro_f1': float(f1.mean()),
        'per_type_f1': {name: float(v) for name, v in zip(groups, f1)},
        'dos_vs_ddos_accuracy': float(exact / same_type) if same_type else 0.0,
        'confusion_matrix': cm_type,
    }


def evaluate_predictions(y_true, proba, classes, run_dir, subtypes=None):
    """subtypes: test satırlarının 19 sınıflı etiketleri; verilirse saldırı türüne göre tespit oranı da raporlanır."""
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

    # İki seviyeli değerlendirme: aynı saldırının DoS ve DDoS sürümü tek tür sayılır (2 sınıfta yok)
    attack_type = attack_type_metrics(cm, classes)
    if attack_type is not None:
        print("🧭 Saldırı türü seviyesi (DoS ve DDoS sürümleri tek tür):")
        print(f"Accuracy:    {attack_type['accuracy']:.4f}")
        print(f"Macro-F1:    {attack_type['macro_f1']:.4f}")
        print(f"DoS/DDoS ayrımı (türü doğru bilinen pencerelerde): {attack_type['dos_vs_ddos_accuracy']:.4f}")
        metrics['attack_type_level'] = {k: v for k, v in attack_type.items() if k != 'confusion_matrix'}
        np.savetxt(os.path.join(run_dir, 'confusion_matrix_attack_type.csv'), attack_type['confusion_matrix'],
                   fmt='%d', delimiter=',', header=','.join(attack_type['groups']), comments='')
        plot_confusion_matrix(attack_type['confusion_matrix'], attack_type['groups'],
                              os.path.join(run_dir, 'confusion_matrix_attack_type.png'))

    # Hangi saldırı türleri Benign sanılıyor: 2 sınıflı modelde bile alt tür bazında görünür
    if subtypes is not None:
        detection = plot_detection_by_subtype(subtypes, np.array(classes)[y_pred] != 'Benign',
                                              os.path.join(run_dir, 'detection_by_attack_type.png'))
        metrics['detection_rate'] = {name: float(rate) for name, rate in detection['flagged_rate'].items()}
        print("Saldırı türüne göre tespit oranı (Benign satırı: yanlış alarm oranı):")
        print(detection['flagged_rate'].sort_values().to_string(float_format=lambda v: f"{v:.2%}"))

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
    plot_per_class_metrics(metrics['per_class'], classes, os.path.join(run_dir, 'per_class_metrics.png'),
                           attack_type)
    plot_confidence(y_true, proba, os.path.join(run_dir, 'confidence_histogram.png'))
    return metrics
