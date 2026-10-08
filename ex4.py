import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Text representation
texts = ["dog", "tiger", "bird", "truck", "church"]
text = np.eye(5)

# Generator: noise + text -> image
generator = tf.keras.Sequential([
    layers.Input(shape=(105,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64 * 64 * 3, activation="sigmoid"),
    layers.Reshape((64, 64, 3))
])

# Discriminator: image + text -> real/fake
image = layers.Input(shape=(64, 64, 3))
condition = layers.Input(shape=(5,))

x = layers.Flatten()(image)
x = layers.Concatenate()([x, condition])
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

discriminator = tf.keras.Model([image, condition], output)

discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Random noise + text
noise = np.random.normal(0, 1, (5, 100))
fake_images = generator.predict(
    np.concatenate([noise, text], axis=1),
    verbose=0
)

# Train discriminator
real_images = np.random.random((5, 64, 64, 3))

discriminator.train_on_batch(
    [real_images, text],
    np.ones((5, 1))
)

discriminator.train_on_batch(
    [fake_images, text],
    np.zeros((5, 1))
)

print("Generated image shape:", fake_images.shape)
print("Discriminator output:",
      discriminator.predict([fake_images, text], verbose=0))

# Unseen text
new_text = np.array([[1, 0, 0, 0, 0]])
new_noise = np.random.normal(0, 1, (1, 100))

new_image = generator.predict(
    np.concatenate([new_noise, new_text], axis=1),
    verbose=0
)

plt.imshow(new_image[0])
plt.title("Generated image for unseen text")
plt.axis("off")
plt.show()