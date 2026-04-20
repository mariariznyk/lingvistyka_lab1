import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "A_Z Handwritten Data.csv"


def load_data(path):
    if not os.path.exists(path):
        print(f"\n  [ERROR] File not found: {path}")
        print(f"  Please download the dataset from:")
        print(f"  https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format")
        print(f"  Unzip and place 'A_Z Handwritten Data.csv' in the project folder.")
        sys.exit(1)

    print(f"  Reading {path} ...")
    df = pd.read_csv(path, header=0)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    images = pixels.reshape(-1, 28, 28)
    return images, labels


def preprocess(X):
    X = X.astype("float32") / 255.0
    if X.ndim == 3:
        X = X[..., np.newaxis]
    return X


def build_model(num_classes=26):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def plot_history(history, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="Train Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved to {save_path}")


def plot_samples(X, y_true, y_pred, labels, save_path="sample_predictions.png"):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    indices = np.random.choice(len(X), 16, replace=False)
    for i, idx in enumerate(indices):
        ax = axes[i // 8, i % 8]
        ax.imshow(X[idx].squeeze(), cmap="gray")
        true_label = labels[y_true[idx]]
        pred_label = labels[y_pred[idx]]
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"T:{true_label} P:{pred_label}", fontsize=9, color=color)
        ax.axis("off")
    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Sample predictions saved to {save_path}")


def main():
    print("  A-Z Handwritten Letters — Letter Recognition with CNN")

    num_classes = 26
    class_labels = [chr(ord("A") + i) for i in range(num_classes)]

    print("\n[1/6] Loading dataset...")
    images, labels = load_data(DATASET_PATH)
    print(f"  Total samples : {images.shape[0]}")
    print(f"  Image shape   : {images.shape[1]}x{images.shape[2]}")
    print(f"  Classes       : {num_classes} (A-Z)")

    print("\n[2/6] Preprocessing and splitting data...")
    print("  Splitting: 70% train / 15% validation / 15% test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

    print(f"  Train set      : {X_train.shape[0]} samples")
    print(f"  Validation set : {X_val.shape[0]} samples")
    print(f"  Test set       : {X_test.shape[0]} samples")

    print("\n[3/6] Building CNN model...")
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\n[4/6] Training the model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[5/6] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    print("\n[6/6] Saving plots...")
    plot_history(history)
    plot_confusion_matrix(y_test, y_pred, class_labels)
    plot_samples(X_test, y_test, y_pred, class_labels)

    model.save("letter_recognition_model.keras")
    print("[OK] Model saved to letter_recognition_model.keras")

    print(f"  DONE! Test accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()