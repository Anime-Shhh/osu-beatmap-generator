import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import prepare_data

# Load preprocessed data
X_audio, X_difficulty, y_hit_objects = prepare_data()

# Debug: Check shapes
print("X_audio length:", len(X_audio))
for i, x in enumerate(X_audio[:5]):  # Print first 5 for inspection
    print(f"X_audio[{i}] shape: {np.array(x).shape}")
print("X_difficulty length:", len(X_difficulty))
print("y_hit_objects length:", len(y_hit_objects))

# Pad sequences to the same length
max_len = max(len(x) for x in X_audio if len(x) > 0)  # Avoid empty arrays
X_audio_padded = tf.keras.preprocessing.sequence.pad_sequences(
    [x for x in X_audio if len(x) > 0], maxlen=max_len, padding='post', dtype='float32'
)
X_difficulty_padded = tf.keras.preprocessing.sequence.pad_sequences(
    [x for x in X_difficulty if len(x) > 0], maxlen=max_len, padding='post', dtype='float32'
)
y_padded = tf.keras.preprocessing.sequence.pad_sequences(
    [y for y in y_hit_objects if len(y) > 0], maxlen=max_len, padding='post', dtype='float32'
)

# Ensure X_audio_padded is 3D (samples, timesteps, features)
if len(X_audio_padded.shape) == 2:  # If 2D, reshape
    X_audio_padded = X_audio_padded[..., np.newaxis]  # Add feature dimension
    X_audio_padded = np.repeat(X_audio_padded, 2, axis=-1)  # Hack: Repeat to match 2 features (temp fix)

# Combine audio and difficulty into one input
X = np.concatenate([X_audio_padded, X_difficulty_padded[..., np.newaxis]], axis=-1)  # Shape: (samples, timesteps, 3)
print("X shape:", X.shape)

# Build the LSTM model
model = tf.keras.Sequential([
    layers.LSTM(128, input_shape=(None, 3), return_sequences=True),  # 3 features: beat_time, energy, OD
    layers.LSTM(64, return_sequences=True),
    layers.Dense(6)  # Output: time, x, y, type, new_combo, hitsound
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X, y_padded, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("osu_generator.h5")
print("Model saved as osu_generator.h5")