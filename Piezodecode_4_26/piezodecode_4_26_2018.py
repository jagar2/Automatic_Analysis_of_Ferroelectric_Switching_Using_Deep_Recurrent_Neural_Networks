import Support_Functions_Savio as sf
from scipy import io
import keras
import numpy as np

embedding = np.load('Piezoresponse_embedding_4_26_2018.npy')
piezoresponse = np.load('Piezoresponse_norm_4_26_2018.npy')



model = keras.models.load_model('./piezodecode_no_drop')
#model = keras.models.load_model('./piezodecode_w_dropout')

seed = 42
np.random.seed(seed)

run_id = 'no_drop'
run_id = sf.check_folder_exist(scratch_path + run_id)
sf.Make_folder(run_id)
model_name = run_id + 'start'

filepath = run_id + '/weights.{epoch:08d}-{val_loss:.4f}.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min', period=1)

logger = keras.callbacks.CSVLogger(run_id + '/log.csv', separator=',', append=True)

history = model.fit(np.atleast_2d(embedding), np.atleast_3d(piezoresponse), epochs=100000,
          batch_size=1800, validation_data=(np.atleast_2d(embedding), np.atleast_3d(piezoresponse)),
          callbacks=[checkpoint, logger])
