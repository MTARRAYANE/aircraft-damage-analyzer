import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# Config
# ===============================
batch_size = 32
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)
epochs = 5

train_dir = "aircraft_damage_dataset_v1/train"
valid_dir = "aircraft_damage_dataset_v1/valid"
test_dir  = "aircraft_damage_dataset_v1/test"

# ===============================
# Data Generators
# ===============================
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'   # 🔥 مهم
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# ===============================
# Number of classes
# ===============================
num_classes = train_generator.num_classes
print("Classes:", train_generator.class_indices)

# ===============================
# Model (VGG16)
# ===============================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',  # 🔥 مهم
    metrics=['accuracy']
)

# ===============================
# Train
# ===============================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator
)

# ===============================
# Save model
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/classifier.h5")

print("✅ Model saved!")

# ===============================
# Evaluate
# ===============================
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc:.4f}")