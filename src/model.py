import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score

from evaluation import plot_training_curves


def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = layers.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(512, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)

    encoded = layers.Dense(256, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)

    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder - simetrik yapı
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)

    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)

    decoded = layers.Dense(512, activation='relu')(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded, name='autoencoder')
    encoder = models.Model(inputs=input_layer, outputs=encoded, name='encoder')
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return autoencoder, encoder


def train_autoencoder(X_train, X_val, encoding_dim=32, epochs=30, batch_size=256):
    print(f"\n🧩 Autoencoder eğitiliyor: {X_train.shape[1]} -> {encoding_dim} boyut")
    autoencoder, encoder = build_autoencoder(X_train.shape[1], encoding_dim)
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
        ],
        verbose=2
    )
    return encoder, history.history


def create_lstm_attention_model(input_dim, num_classes, encoder=None, reshape_dim=8, num_heads=4):
    inputs = layers.Input(shape=(input_dim,))

    # Autoencoder varsa ham özellikler önce encoder'dan geçer (dondurulmuş ya da ince ayarlı)
    x = encoder(inputs) if encoder is not None else inputs

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # LSTM'e verilecek diziyi oluştur: reshape_dim adım x (512 / reshape_dim) özellik
    reshaped = layers.Reshape((reshape_dim, 512 // reshape_dim))(x)

    lstm_units = 128 if num_classes > 2 else 64
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True,
                                 kernel_regularizer=l2(0.0005),
                                 recurrent_regularizer=l2(0.0005)))(reshaped)
    lstm_out = layers.BatchNormalization()(lstm_out)
    lstm_out = layers.Dropout(0.2)(lstm_out)

    # Çok başlıklı attention: her başlık zaman adımları üzerinde ayrı bir ağırlık dağılımı öğrenir.
    # Softmax zaman ekseninde (axis=1) alınır; tek elemanlı eksende alınırsa ağırlıklar hep 1 olur.
    attention_heads = []
    for _ in range(num_heads):
        score = layers.Dense(lstm_units, activation='tanh')(lstm_out)
        score = layers.Dense(1)(score)
        weights = layers.Softmax(axis=1)(score)
        context = layers.Multiply()([lstm_out, weights])
        attention_heads.append(layers.GlobalAveragePooling1D()(context))

    attention_concat = layers.Concatenate()(attention_heads)
    attention_concat = layers.BatchNormalization()(attention_concat)
    attention_concat = layers.Dropout(0.2)(attention_concat)

    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(attention_concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # İkili dahil her durumda softmax: etiketler tamsayı, kayıp sparse categorical crossentropy
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class ValidationMonitor(keras.callbacks.Callback):
    """Her epoch sonunda doğrulama setinde macro-F1'i ve sınıf başına F1'i hesaplar; erken durdurma ve checkpoint
    val_macro_f1'i izler. training_log.csv ve eğitim grafikleri her epoch'ta yenilenir, böylece eğitim sürerken
    run klasöründen izlenebilir."""

    def __init__(self, X_val, y_val, classes, run_dir):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.classes = classes
        self.run_dir = run_dir
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, batch_size=4096, verbose=0).argmax(axis=1)
        f1 = f1_score(self.y_val, y_pred, labels=list(range(len(self.classes))), average=None, zero_division=0)
        logs['val_macro_f1'] = float(f1.mean())
        print(f"  val_macro_f1: {logs['val_macro_f1']:.4f}")

        # Öğrenme oranı, ReduceLROnPlateau bu epoch'un sonunda değiştirmeden önce okunur
        row = {'epoch': epoch + 1, 'lr': float(keras.backend.get_value(self.model.optimizer.learning_rate))}
        row.update({k: float(logs[k]) for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy', 'val_macro_f1')})
        row.update({f'val_f1_{name}': float(v) for name, v in zip(self.classes, f1)})
        self.rows.append(row)

        log = pd.DataFrame(self.rows)
        log.to_csv(os.path.join(self.run_dir, 'training_log.csv'), index=False)
        plot_training_curves(log, self.classes, self.run_dir)


def train_model(model, X_train, y_train, X_val, y_val, run_dir, classes, epochs=50, batch_size=512,
                class_weights=None):
    print(f"\n🚀 Eğitim başlatılıyor: {len(X_train)} örnek, batch_size={batch_size}")
    # Çok kısa eğitimde BatchNorm'un hareketli ortalamaları oturmaz; model tahmin modunda tek sınıfa çökebilir
    steps = int(np.ceil(len(X_train) / batch_size)) * epochs
    if steps < 300:
        print(f"⚠️ Toplam {steps} eğitim adımı çok az: BatchNorm istatistikleri oturmadan model tahmin modunda "
              f"tek sınıfa çökebilir. En az ~300 adım önerilir (daha fazla epoch ya da veri).")

    best_weights = os.path.join(run_dir, 'best_weights.h5')
    callbacks = [
        ValidationMonitor(X_val, y_val, classes, run_dir),
        keras.callbacks.EarlyStopping(
            monitor='val_macro_f1',
            mode='max',
            patience=8,
            min_delta=1e-4
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            best_weights,
            monitor='val_macro_f1',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )

    # Erken durma tetiklenmese de en iyi epoch'un ağırlıklarına dön
    model.load_weights(best_weights)
    return model, history
