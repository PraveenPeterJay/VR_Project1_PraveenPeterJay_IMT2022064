import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator





warnings.filterwarnings('ignore')

def load_dataset(dataset_path):
    with_mask_path = os.path.join(dataset_path, 'with_mask')
    without_mask_path = os.path.join(dataset_path, 'without_mask')
    
    images = []
    labels = []
    
 
    for img_name in os.listdir(with_mask_path):
        img_path = os.path.join(with_mask_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)
                labels.append(1)  # 1 for with_mask
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
   
    for img_name in os.listdir(without_mask_path):
        img_path = os.path.join(without_mask_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)
                labels.append(0)  # 0 for without_mask
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def extract_handcrafted_features(images):
    features = []
    
    for img in images:
        feature_vector = []
        
      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. HOG features
        hog = extract_hog_features(gray)
        feature_vector.extend(hog)
        
        # 2. Color histograms (per channel)
        hist_features = extract_color_histograms(img)
        feature_vector.extend(hist_features)
        
        # 3. Statistical features
        stats_features = extract_statistical_features(img)
        feature_vector.extend(stats_features)
        features.append(feature_vector)
    
    return np.array(features)

def extract_hog_features(gray_img, win_size=(32, 32), block_size=(16, 16), 
                         block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(gray_img)
    return h.flatten()

def extract_color_histograms(img, bins=32):
    hist_features = []
    
    # Loop over each channel
    for i in range(3):
        channel = img[:, :, i]
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    return hist_features

def extract_statistical_features(img):
    stats = []
    
    # For each channel
    for i in range(3):
        channel = img[:, :, i]
        # Mean and standard deviation
        mean, std = cv2.meanStdDev(channel)
        stats.extend([mean[0][0], std[0][0]])
        
        # Add more statistics
        stats.append(np.median(channel))
        stats.append(np.max(channel) - np.min(channel))  # Range
        
    return stats



def build_neural_network(input_shape):
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # SVM
    print("Training SVM...")
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM - Accuracy: {svm_accuracy:.2f}")
    with open('accuracy_report.txt', 'a') as f:  # Append accuracy to the file
            f.write(f"Model SVM - Accuracy: {svm_accuracy:.2f}\n")
    print("Building and training neural network...")
    # Build model
    model = build_neural_network(X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    with open('accuracy_report.txt', 'a') as f:  # Append accuracy to the file
            f.write(f"Artificial Neural Network Model - Accuracy: {accuracy:.2f}\n")
    # Classification report
    with open("classification_report.txt", "w") as f:
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Without Mask', 'With Mask'],
                yticklabels=['Without Mask', 'With Mask'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_ANN.png")
    plt.close()
    
    # Save training history plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("model_accuracy_ANN.png")
    plt.close()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("model_loss_ANN.png")
    plt.close()
    
    return model



def cnn(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Best model
    best_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    history1 = best_model.fit(datagen.flow(X_train, y_train, batch_size=64), 
                               epochs=50, 
                               validation_data=(X_test, y_test),
                               callbacks=[early_stopping],
                               verbose=1)

    # Model 1 with one less convolution layer
    model1 = Sequential([
        Conv2D(32, (3, 3), activation='tanh', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), loss='binary_crossentropy', metrics=['accuracy'])
    history2 = model1.fit(datagen.flow(X_train, y_train, batch_size=64), 
                           epochs=50, 
                           validation_data=(X_test, y_test),
                           callbacks=[early_stopping],
                           verbose=1)

    # Best model with different optimizer and batch size
    best_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    history3 = best_model.fit(datagen.flow(X_train, y_train, batch_size=128), 
                               epochs=50, 
                               validation_data=(X_test, y_test),
                               callbacks=[early_stopping],
                               verbose=1)

    # Classification report for all models

    for idx, (model, history) in enumerate([(best_model, history1), (model1, history2), (best_model, history3)]):
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        model_accuracy = accuracy_score(y_test, y_pred)
        with open(f'classification_report_model_{idx+1}.txt', 'w') as f:
            f.write(f"\nClassification Report for Model {idx+1}:\n")
            f.write(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
        
        with open('accuracy_report.txt', 'a') as f:  # Append accuracy to the file
            f.write(f"Model {idx+1} - Accuracy: {model_accuracy:.2f}\n")
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Without Mask', 'With Mask'],
                    yticklabels=['Without Mask', 'With Mask'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix for Model {idx+1}')
        plt.savefig(f'confusion_matrix_model_{idx+1}.png')
        plt.close()

    # Save training history plots
    plt.figure(figsize=(12, 4))
    plt.plot(history1.history['accuracy'], label='Best Model')
    plt.plot(history2.history['accuracy'], label='Model 1')
    plt.plot(history3.history['accuracy'], label='Best Model (Adagrad, Batch 128)')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('model_accuracy.png')
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(history1.history['loss'], label='Best Model')
    plt.plot(history2.history['loss'], label='Model 1')
    plt.plot(history3.history['loss'], label='Best Model (Adagrad, Batch 128)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('model_loss.png')
    plt.close()
def ml_classifier(images,labels):
    features = extract_handcrafted_features(images)
    print(f"Extracted features with shape: {features.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
def main():
    dataset_path = "dataset"  # Update this with your local path to the dataset
    
    
    images, labels = load_dataset(dataset_path)
    print(f"Loaded {len(images)} images, with {np.sum(labels)} with masks and {len(labels) - np.sum(labels)} without masks")
    print(images.shape)
    
    # part A
    
    #ml_classifier(images,labels)
    # part B
    cnn(images,labels)
    

    
if __name__ == "__main__":
    main()