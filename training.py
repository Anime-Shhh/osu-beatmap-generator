import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import prepare_data

# Load preprocessed data
X_audio, X_difficulty, y_hit_objects = prepare_data()

# Filter out empty arrays and debug
X_audio = [x for x in X_audio if len(x) > 0]
X_difficulty = [x for x in X_difficulty if len(x) > 0]
y_hit_objects = [y for y in y_hit_objects if len(y) > 0]

print("X_audio length after filter:", len(X_audio))
for i, x in enumerate(X_audio[:5]):
    print(f"X_audio[{i}] shape: {np.array(x).shape}")
print("X_difficulty length:", len(X_difficulty))
print("y_hit_objects length:", len(y_hit_objects))

if not X_audio:
    raise ValueError("No valid data to train on after filtering empty arrays")

# Pad sequences to the same length
max_len = max(len(x) for x in X_audio)
X_audio_padded = tf.keras.preprocessing.sequence.pad_sequences(X_audio, maxlen=max_len, padding='post', dtype='float32')
X_difficulty_padded = tf.keras.preprocessing.sequence.pad_sequences(X_difficulty, maxlen=max_len, padding='post', dtype='float32')
y_padded = tf.keras.preprocessing.sequence.pad_sequences(y_hit_objects, maxlen=max_len, padding='post', dtype='float32')

# Combine audio and difficulty into one input
X = np.concatenate([X_audio_padded, X_difficulty_padded[..., np.newaxis]], axis=-1)
print("X shape:", X.shape)

# Build the LSTM model
model = tf.keras.Sequential([
    layers.LSTM(128, input_shape=(None, 3), return_sequences=True),
    layers.LSTM(64, return_sequences=True),
    layers.Dense(6)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X, y_padded, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("osu_generator.h5")
print("Model saved as osu_generator.h5")