import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASSES = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]

def load_and_preprocess_data():
    
    (trainX, trainY), (testX, testY) = datasets.cifar10.load_data()
    
    
    trainY = trainY.reshape(-1)
    testY = testY.reshape(-1)
    
    
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    return (trainX, trainY), (testX, testY)

def plot_sample(X, y, index):
    plt.figure(figsize=(5,5))
    plt.imshow(X[index])
    plt.xlabel(CLASSES[y[index]], fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),  
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  
        layers.Dropout(0.2),  
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
   
    (trainX, trainY), (testX, testY) = load_and_preprocess_data()
    
   
    datagen = create_data_generator()
    datagen.fit(trainX)
    
    
    model = build_cnn_model()
    history = model.fit(
        datagen.flow(trainX, trainY, batch_size=32),
        epochs=20,
        validation_data=(testX, testY),
        verbose=1
    )
    
   
    test_loss, test_accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    
    y_pred = model.predict(testX)
    y_classes = np.argmax(y_pred, axis=1)
    
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"True: {CLASSES[testY[i]]}, Predicted: {CLASSES[y_classes[i]]}")
        plot_sample(testX, testY, i)

if __name__ == "__main__":
    main()