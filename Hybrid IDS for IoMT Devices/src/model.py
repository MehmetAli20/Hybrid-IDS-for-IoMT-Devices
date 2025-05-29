import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_lstm_attention_model(input_dim, num_classes, sequence_len=4, reshape_dim=None):
    # AE'den gelen encoded feature'ı (ör. 32) bir "sequence" olarak LSTM'ye vermek için reshape edelim
    if reshape_dim is None:
        # Mümkünse kare/küçük bir çarpan
        for i in range(2, input_dim+1):
            if input_dim % i == 0:
                reshape_dim = i
                break
        sequence_len = input_dim // reshape_dim
    else:
        sequence_len = input_dim // reshape_dim

    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(reshape_dim*sequence_len, activation='relu')(inputs)
    x = layers.Reshape((sequence_len, reshape_dim))(x)
    # LSTM + Attention
    lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    attention = layers.Dense(1, activation='tanh')(lstm1)
    attention = layers.Softmax(axis=1)(attention)
    x = layers.Multiply()([lstm1, attention])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=metrics
    )
    print("✅ Model oluşturuldu")
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=512):
    # Class weights hesapla - daha agresif dengeleme
    class_weights = {}
    y_integers = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_integers)
    total_samples = len(y_integers)
    max_count = np.max(class_counts)
    
    for i in range(len(class_counts)):
        # Daha agresif dengeleme için kare kök kullan
        class_weights[i] = np.sqrt(max_count / class_counts[i])
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,  # Daha uzun patience
            min_delta=0.0001,  # Daha hassas
            restore_best_weights=True,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Daha agresif azalma
            patience=40,  # Learning rate azaltma için daha uzun bekleme
            min_lr=0.000001,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    print(f"🚀 Eğitim başlatılıyor: {len(X_train)} örnek, batch_size={batch_size}")
    
    # Gradient clipping eklendi
    optimizer = model.optimizer
    optimizer.clipnorm = 1.0
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history