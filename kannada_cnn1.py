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

# Data Augmentation

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compile a model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Train the model

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                              epochs=5,
                              validation_data=(x_test, y_test),
                              verbose=2,
                              steps_per_epoch=x_train.shape[0] // 128
                              )


# Evaluate the model
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# save the model
model.save('knr_dataug.model')
