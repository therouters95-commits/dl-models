import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model


url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

archive = tf.keras.utils.get_file(
    "imagenette2-160.tgz",
    url,
    untar=True
)
# Load 10 images
dataset = tf.keras.utils.image_dataset_from_directory(
    archive + "/imagenette2-160/val",
    image_size=(224, 224),
    batch_size=10,
    shuffle=True
)

images, labels = next(iter(dataset))

# ImageNet class names
names = {
    "n01440764": "fish",
    "n02102040": "dog",
    "n02979186": "cassette player",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute"
}

# Create captions
captions = []

for label in labels:
    name = names[dataset.class_names[label]]
    captions.append("start a photo of a " + name + " end")

print("Example:", captions[0])

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(x) for x in sequences)

print("Vocabulary size:", vocab_size)

# CNN feature extraction
cnn = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

cnn.trainable = False

features = cnn.predict(
    preprocess_input(images),
    verbose=0
)

print("Feature shape:", features.shape)

# Create training examples
X_image = []
X_text = []
Y = []

for i in range(len(sequences)):

    seq = sequences[i]

    for j in range(1, len(seq)):

        X_image.append(features[i])

        X_text.append(
            seq[:j]
        )

        Y.append(
            seq[j]
        )

X_image = np.array(X_image)

X_text = pad_sequences(
    X_text,
    maxlen=max_length,
    padding="pre"
)

Y = np.array(Y)

# LSTM decoder
image_input = Input(shape=(512,))
text_input = Input(shape=(max_length,))

image = Dense(128, activation="tanh")(image_input)

text = Embedding(
    vocab_size,
    128
)(text_input)

lstm = LSTM(128)(text, initial_state=[image, image])

output = Dense(
    vocab_size,
    activation="softmax"
)(lstm)

model = Model(
    [image_input, text_input],
    output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    [X_image, X_text],
    Y,
    epochs=100,
    verbose=0
)

print("Training complete")

# Generate caption
feature = features[0:1]

words = ["start"]

for i in range(max_length):

    sequence = tokenizer.texts_to_sequences(
        [" ".join(words)]
    )

    sequence = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="pre"
    )

    prediction = model.predict(
        [feature, sequence],
        verbose=0
    )

    word_id = np.argmax(prediction[0])

    word = tokenizer.index_word.get(
        word_id,
        ""
    )

    if word == "end" or word == "":
        break

    words.append(word)

caption = " ".join(words[1:])

print("Generated Caption:", caption)

# Display image
plt.imshow(
    images[0].numpy().astype("uint8")
)

plt.title(
    caption
)

plt.axis("off")
plt.show()