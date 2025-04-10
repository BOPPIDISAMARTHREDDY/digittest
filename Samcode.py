import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Load and preprocess data
df = pd.read_csv(r"//")

# Ensure label column is numeric and clean data
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna()

# Separate features and labels
X = df.drop('label', axis=1).values.astype('float32') / 255.0
y = df['label'].astype(int).values

# Reshape for CNN input
X = X.reshape(-1, 28, 28, 1)

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, 10)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 5. Save the model
model.save(r"model path")

print("✅ Model training complete and saved to 'samreddy pragament\\best_model.h5'")
