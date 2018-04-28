import dataset
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv3D

pixel_size = 32

batch_size = 128
num_classes = 2
epochs = 500

x_train, x_test, y_train, y_test = dataset.train_test_dataset(pixel_size)

# just try 1
x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')
#x_train /= 255 # or 
x_train = x_train/128 - 1.0
#x_test /= 255 # or 
x_test = x_test/128 - 1.0

print(x_train.shape)
print(x_train.shape[1:])

# From keras/cifar10_cnn example

model = Sequential()
model.add(Conv2D(96, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization()) ###
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization()) ###
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization()) ###
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
#opt = keras.optimizers.Adadelta()

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

''''''


# Feed data to the model

'''
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
'''

#'''
datagen = ImageDataGenerator(rotation_range=20, \
				width_shift_range=0.2, height_shift_range=0.2, \
				horizontal_flip=True) # real-time data augmentation

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=epochs)
#'''

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

