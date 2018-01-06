from __future__ import division, print_function
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import os
import time
import dataset2

train_path = 'training_data'

classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)

data = dataset2.read_train_sets2(train_path, classes, validation_size=0.3)

Xtrain = data.train.images
Ytrain = data.train.labels
Xtest = data.valid.images
Ytest = data.valid.labels

# ============================================== #

DIM1 = 28
DIM2 = 28
INPUT_DIM = 1568

input_signal = Input(shape=(DIM1, DIM2, 2))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_signal)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# ============================================== #

model = Model(inputs=input_signal, outputs=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

model.fit(Xtrain, Ytrain, epochs=50, batch_size=128, shuffle=True, validation_data=(Xtest, Ytest))

# ============================================== #

print("")
sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
sdr.err_ppm = 56
sdr.gain = 'auto'

correct_predictions = 0


def read_samples(freq):
    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(sample_rate * 0.25)  # sample 1/4 sec
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


def check(freq, corr):
    samples = []
    iq_samples = read_samples(freq)
    iq_samples = signal.decimate(iq_samples, 48, zero_phase=True)

    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    # iq_samples = np.concatenate((real, imag))
    # iq_samples = np.reshape(iq_samples, (-1, 25000))

    iq_samples = np.ravel(np.column_stack((real, imag)))
    iq_samples = iq_samples[:INPUT_DIM]

    samples.append(iq_samples)

    samples = np.array(samples)

    # reshape for convolutional model
    samples = np.reshape(samples, (len(samples), DIM1, DIM2, 2))

    prediction = model.predict(samples)

    # print predicted label
    maxim = 0.0
    maxlabel = ""
    for sigtype, probability in zip(classes, prediction[0]):
        if probability >= maxim:
            maxim = probability
            maxlabel = sigtype
    print(freq / 1000000, maxlabel, maxim * 100)

    # calculate validation percent
    if corr == maxlabel:
        global correct_predictions
        correct_predictions += 1


check(92900000, "wfm")
check(49250000, "tv")
check(95000000, "wfm")
check(104000000, "wfm")
check(422600000, "tetra")
check(100500000, "wfm")
check(120000000, "other")
check(106300000, "wfm")
check(942200000, "gsm")
check(107800000, "wfm")

sdr.close()

print("Validation:", correct_predictions / 10 * 100)
