import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import prepare_data

class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss'):.4f}, val_loss = {logs.get('val_loss'):.4f}, "
              f"hit_objects_loss = {logs.get('hit_objects_loss'):.4f}, obj_types_loss = {logs.get('obj_types_loss'):.4f}")

X_audio, X_difficulty, y_hit_objects, y_obj_types = prepare_data()

X_audio = [x for x in X_audio if len(x) > 0]
X_difficulty = [x for x in X_difficulty if len(x) > 0]
y_hit_objects = [y for y in y_hit_objects if len(y) > 0]
y_obj_types = [y for y in y_obj_types if len(y) > 0]

print("X_audio length after filter:", len(X_audio))
for i, x in enumerate(X_audio[:5]):
    print(f"X_audio[{i}] shape: {np.array(x).shape}")
print("X_difficulty length:", len(X_difficulty))
print("y_hit_objects length:", len(y_hit_objects))
print("y_obj_types length:", len(y_obj_types))

if not X_audio:
    raise ValueError("No valid data to train on after filtering empty arrays")

max_len = min(max(len(x) for x in X_audio), 500)
X_audio_padded = tf.keras.preprocessing.sequence.pad_sequences(X_audio, maxlen=max_len, padding='post', dtype='float32')
X_difficulty_padded = tf.keras.preprocessing.sequence.pad_sequences(X_difficulty, maxlen=max_len, padding='post', dtype='float32')
y_hit_objects_padded = tf.keras.preprocessing.sequence.pad_sequences(y_hit_objects, maxlen=max_len, padding='post', dtype='float32')
y_obj_types_padded = tf.keras.preprocessing.sequence.pad_sequences(y_obj_types, maxlen=max_len, padding='post', dtype='int32')

X = np.concatenate([X_audio_padded, X_difficulty_padded[..., np.newaxis]], axis=-1)
print("X shape:", X.shape)
print(f"X sample: {X[0, :5]}")
print(f"y_hit_objects sample: {y_hit_objects_padded[0, :5]}")
print(f"y_obj_types sample: {y_obj_types_padded[0, :5]}")

inputs = tf.keras.Input(shape=(None, X.shape[-1]))
x = layers.LSTM(64, return_sequences=True)(inputs)  # No Masking
x = layers.Dropout(0.2)(x)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
hit_objects = layers.Dense(5, name='hit_objects')(x)
obj_types = layers.Dense(3, activation='softmax', name='obj_types')(x)

model = tf.keras.Model(inputs=inputs, outputs=[hit_objects, obj_types])
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss={
        'hit_objects': tf.keras.losses.MeanSquaredError(),
        'obj_types': tf.keras.losses.SparseCategoricalCrossentropy()
    },
    loss_weights={'hit_objects': 0.5, 'obj_types': 1.5}
)
model.summary()

# Early stopping to protect accuracy
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='hit_objects_loss',
    patience=5,
    mode='min',
    min_delta=0.01,
    restore_best_weights=True
)

model.fit(
    X,
    {'hit_objects': y_hit_objects_padded, 'obj_types': y_obj_types_padded},
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[PrintLossCallback(), early_stopping]
)

model.save("osu_generator_advanced.h5")
print("Model saved as osu_generator_advanced.h5")