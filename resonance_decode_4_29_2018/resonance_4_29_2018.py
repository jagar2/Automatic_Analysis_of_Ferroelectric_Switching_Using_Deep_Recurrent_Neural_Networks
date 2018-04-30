import Support_Functions_Savio as sf
from scipy import io
import keras
import numpy as np

embedding = np.load('decode_w_drop_embed.npy')
resonance = np.load('resonance_4_29_2018.npy')



model = keras.models.load_model('./decode_wo_drop')
#model = keras.models.load_model('./decode_w_drop)

seed = 42
np.random.seed(seed)

scratch_path = '/global/scratch/jagar/resonance_decode_4_29_2018/'

run_id = 'no_drop'
run_id = sf.check_folder_exist(scratch_path + run_id)
sf.Make_folder(run_id)
model_name = run_id + 'start'

filepath = run_id + '/weights.{epoch:08d}-{val_loss:.4f}.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min', period=1)

logger = keras.callbacks.CSVLogger(run_id + '/log.csv', separator=',', append=True)

history = model.fit(np.atleast_2d(embedding), np.atleast_3d(resonance), epochs=100000,
          batch_size=1800, validation_data=(np.atleast_2d(embedding), np.atleast_3d(resonance)),
          callbacks=[checkpoint, logger])
