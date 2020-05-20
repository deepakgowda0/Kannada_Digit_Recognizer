import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

training_images = pd.read_csv('Kannada-MNIST/train.csv')
test_images = pd.read_csv('Kannada-MNIST/test.csv')

training_labels = training_images['label']

training_images = training_images.drop('label', axis=1)
test_images = test_images.drop('id', axis=1)

x_train, x_test, y_train, y_test = train_test_split(training_images, training_labels, test_size=0.3)

# Shapes of train and test data

# print(x_train.shape) # (42000, 784)
# print(x_test.shape)  # (18000, 784)

# print(y_train.shape) # (42000,)
# print(y_test.shape)  # (18000,)

# Resize and Normalize

x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test / 255.0

# print(x_train.shape)  # (42000, 28, 28, 1)
# print(x_test.shape)   # (18000, 28, 28, 1)


# Build a CNN model

model = tf.keras.models.Sequential([
    # Convolution layers
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # Flatten Layer
    tf.keras.layers.Flatten(),

    # Dense Layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')

])

# Compile a model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)

# save the model
model.save('kannada_number_reader.model')

