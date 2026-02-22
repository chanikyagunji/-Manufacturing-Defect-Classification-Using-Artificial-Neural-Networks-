import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory

# -----------------------------
# Load Dataset
# -----------------------------
train_ds = image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32
)

# Preprocess input for ResNet
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# -----------------------------
# Load Pretrained ResNet50
# -----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# -----------------------------
# Add Custom Layers
# -----------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(val_ds)
print("Validation Accuracy:", acc)
