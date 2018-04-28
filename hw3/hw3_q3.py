import dataset
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

pixel_size = 32

batch_size = 128
num_classes = 2
epochs = 300
learning_rate = 0.01

x_train, x_test, y_train, y_test = dataset.train_test_dataset(pixel_size)

x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')

x_train = x_train/128 - 1.0
x_test = x_test/128 - 1.0

print(x_train.shape)
print(x_train.shape[1:])

# modified https://github.com/f00-/mnist-lenet-keras/lenet.py

model = Sequential()

model.add(Conv2D(6, (5, 5), padding='same', \
			input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=learning_rate)

model.compile(loss='categorical_crossentropy', \
		optimizer=opt, \
		metrics=['accuracy'])

# Feed data

datagen = ImageDataGenerator(rotation_range=20, \
				width_shift_range=0.2, \
				height_shift_range=0.2, \
				horizontal_flip=True) # real-time data augmentation

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])
