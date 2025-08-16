# Al-Future-Directions-
# Edge AI & IoT Project

## Overview
This repository contains the practical implementation of an Edge AI and AI-driven IoT project, along with ethical and futuristic AI proposals. The project is divided into three main parts:

- **Part 2: Practical Implementation**
  - **Task 1:** Edge AI Prototype (Image Classification for Recyclables)
  - **Task 2:** AI-Driven IoT Concept (Smart Agriculture)
  - **Task 3:** Ethics in Personalized Medicine
- **Part 3: Futuristic AI Proposal (2030)**

---

## Repository Structure


---

## Part 2: Practical Implementation

### Task 1: Edge AI Prototype
- **Goal:** Train a lightweight image classification model for recyclable items and deploy it using TensorFlow Lite.
- **Files:**
  - `recycle_classifier.ipynb` → Notebook with training, evaluation, and TFLite conversion.
  - `recycle_model.h5` → Trained Keras model.
  - `recycle_model.tflite` → Converted TensorFlow Lite model.
- **Instructions:** Open the notebook and run each cell. Modify dataset paths as needed. Sample images are included for testing inference.

### Task 2: Smart Agriculture Proposal
- **Goal:** AI-driven IoT system to predict crop yields.
- **Files:**
  - `proposal.pdf` → 1-page proposal with AI workflow.
  - `diagram.png` → Data flow diagram.
  - `sensors_list.md` → List of sensors used.

---

## **2️⃣ Jupyter Notebook Skeleton (`recycle_classifier.ipynb`)**

```python
# Edge AI Prototype: Recyclable Items Classification

# -----------------------------
# 1. Import Libraries
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 2. Load Dataset
# -----------------------------
# TODO: Replace with actual dataset path
train_dir = "data/train"
test_dir = "data/test"

# Example: Using ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# -----------------------------
# 3. Build Lightweight CNN Model
# -----------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 4. Train Model
# -----------------------------
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# 6. Convert to TensorFlow Lite
# -----------------------------
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
with open("recycle_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")

# -----------------------------
# 7. Test TFLite Model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test first image from test_generator
sample_img, sample_label = next(test_generator)
interpreter.set_tensor(input_details[0]['index'], sample_img[:1])
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])
print("Predicted Class:", np.argmax(pred))
print("True Class:", np.argmax(sample_label[0]))

### Task 3: Ethics in Personalized Medicine
- **Goal:** Identify biases in AI-based treatment recommendations.
- **Files:**
  - `ethics_analysis.pdf` → 300-word analysis on fairness strategies.

---

## Part 3: Futuristic AI Proposal (2030)
- **Goal:** Propose a novel AI application for 2030.
- **Files:**
  - `AI_2030_Concept.pdf` → Concept paper describing problem, AI workflow, and societal risks/benefits.

---

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

Install dependencies:
```bash
pip install -r requirements.txt

