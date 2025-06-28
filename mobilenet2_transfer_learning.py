from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- (Optional) Explicitly set a backend if plots still don't show after fix ---
# import matplotlib
# matplotlib.use('TkAgg') # Try 'TkAgg' or 'Qt5Agg'

# ========== CONFIGURATION ==========
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4 # adjust if you have different classes

TRAIN_DIR = 'dataset/train' # adjust path
VALID_DIR = 'dataset/validation' # adjust path
TEST_DIR = 'dataset/validation' # Often, the validation set is also used as a test set for final evaluation if no separate test set exists. Adjust if you have a dedicated 'test' folder.
MODEL_SAVE_PATH = 'model/trained_model.h5'
PLOTS_SAVE_DIR = 'plots' # NEW: Directory to save plots

# Create the plots directory if it doesn't exist
if not os.path.exists(PLOTS_SAVE_DIR):
    os.makedirs(PLOTS_SAVE_DIR)

# ========== DATA AUGMENTATION ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255) # For validation
test_datagen = ImageDataGenerator(rescale=1./255) # For evaluation/prediction

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Create a generator for the test/evaluation set
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, # Using VALID_DIR for demonstration. Replace with your actual TEST_DIR if separate.
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # IMPORTANT: Set shuffle to False for test/evaluation set to maintain order for true labels
)

print("Class Indices from training:", train_generator.class_indices)

# ========== BASE MODEL ==========
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False # freeze the base model

# ========== CUSTOM HEAD ==========
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ========== COMPILE ==========
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ========== TRAIN ==========
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ========== EVALUATE AND GENERATE PLOTS ==========

# Get predictions on the test/evaluation set
test_steps_per_epoch = np.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator, steps=int(test_steps_per_epoch)) # Changed to int()

# Get predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get true class labels
true_classes = test_generator.classes

# Get class labels in the order used by the generator
labels_map = test_generator.class_indices
labels = list(labels_map.keys())
class_labels = [k for k, v in sorted(labels_map.items(), key=lambda item: item[1])]


# For the Accuracy Plot:
plt.figure()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(PLOTS_SAVE_DIR, 'model_accuracy.png')) # NEW: Save plot
plt.show(block=False) # Allows the script to continue without closing this plot


# For the Loss Plot:
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(os.path.join(PLOTS_SAVE_DIR, 'model_loss.png')) # NEW: Save plot
plt.show(block=False) # Allows the script to continue without closing this plot


# For the Confusion Matrix Plot:
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(PLOTS_SAVE_DIR, 'confusion_matrix.png')) # NEW: Save plot
plt.show(block=True) # This will block execution until you close the window, ensuring you see it.


# ========== PRINT CLASSIFICATION REPORT AND ACCURACY ==========
print("\n--- Classification Report ---")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

overall_accuracy = accuracy_score(true_classes, predicted_classes) * 100
print(f"\nOverall Accuracy of the Model: {overall_accuracy:.1f}%")

# ========== SAVE ==========
if not os.path.exists("model"):
    os.makedirs("model")
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")