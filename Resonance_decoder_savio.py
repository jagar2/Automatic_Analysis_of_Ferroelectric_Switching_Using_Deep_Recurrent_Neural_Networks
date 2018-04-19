import Support_Functions_Savio as sf
from scipy import io
import keras
import numpy as np

import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.layers.core import Lambda
from keras.optimizers import Adam



Resonance = np.load('resonance.npy')
low_d = np.load('resonance_low_d.npy')

model_decode_Reson = Sequential()
model_decode_Reson.add(BatchNormalization(input_shape=(7,)))
model_decode_Reson.add(RepeatVector(96))
model_decode_Reson.add(Bidirectional(LSTM(256, return_sequences=True)))
model_decode_Reson.add(Bidirectional(LSTM(256, return_sequences=True)))
model_decode_Reson.add(TimeDistributed(Dense(1, activation='linear')))
model_decode_Reson.compile(Adam(3e-5), loss='mse')

seed = 42
np.random.seed(seed)

scratch_path = '/global/scratch/jagar/decode_resonance'

run_id = ''

run_id = sf.check_folder_exist(scratch_path + run_id)

sf.Make_folder(run_id)

model_name = run_id + 'start'
keras.models.save_model(model_decode_Reson, run_id + '/start_seed_{0:03d}.h5'.format(seed))

filepath = run_id + '/weights.{epoch:07d}-{val_loss:.6f}.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min', period=1)

logger = keras.callbacks.CSVLogger(run_id + '/log.csv', separator=',', append=True)

history = model_decode_Reson.fit(np.atleast_2d(low_d), np.atleast_3d(Resonance), epochs=1000000,
          batch_size=3600, validation_data=(np.atleast_2d(low_d), np.atleast_3d(Resonance)),
          callbacks=[checkpoint, logger])
