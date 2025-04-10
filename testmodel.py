import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


model = tf.keras.models.load_model(r"model path")


print("Loading dataset (with headers)...")
df = pd.read_csv(r"data set path")


df['label'] = pd.to_numeric(df['label'], errors='coerce')


print("Label column values:", df['label'].unique())


fig, axs = plt.subplots(3, 4, figsize=(12, 9))
fig.suptitle("Actual vs Predicted Digits (MNIST)", fontsize=16)
axs = axs.flatten()


for digit in range(10):
    sample = df[df['label'] == digit].head(1)

    if sample.empty:
        print(f"No sample found for digit: {digit}")
        continue

    row = sample.iloc[0]
    label = int(row['label'])
    pixels = row.iloc[1:].astype(np.uint8).values
    image = pixels.reshape(28, 28)

    img_input = image / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)


    prediction = model.predict(img_input, verbose=0)
    predicted_digit = np.argmax(prediction)


    axs[digit].imshow(image, cmap='gray')
    axs[digit].set_title(f"Actual: {label} | Pred: {predicted_digit}")
    axs[digit].axis('off')


for i in range(10, len(axs)):
    axs[i].axis('off')

plt.tight_layout()
plt.show()
