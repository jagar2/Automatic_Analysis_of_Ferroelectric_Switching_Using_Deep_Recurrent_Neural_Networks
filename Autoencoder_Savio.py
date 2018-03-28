import Support_Functions_Savio as sf
from scipy import io
import keras
import numpy as np

## loads the data file
#data = io.matlab.loadmat('./Data.mat')
#
## extracts the voltage vector
#voltage = data['Voltagedata_mixed']
#
## extracts the hyteresis loopsb
#Piezoresponse = data['Loopdata_mixed']
#Amplitude = data['OutA2_mixed']
#Phase = data['OutPhi1_mixed']
#Resonance = data['Outw2_mixed']
#Quality = data['OutQ2_mixed']
#
## interpolates missing points
#Piezoresponse = sf.interpolate_missing_points(Piezoresponse, 'linear')
#Amplitude = sf.clean_and_interpolate(Amplitude, 'linear')
#Phase = sf.clean_and_interpolate(Phase, 'linear')
#Resonance = sf.clean_and_interpolate(Resonance, 'linear')
#Quality = sf.clean_and_interpolate(Quality, 'linear')
#
#Resonance = sf.sg_filter_data(Resonance, fit_type='linear')
#Resonance = sf.normalize_data(Resonance)

Resonance = np.load('resonance.npy')

model, run_id = sf.rnn_auto('lstm', size=64, num_encode_layers = 4, num_decode_layers = 4,
                                        embedding = 16, n_step = 96, lr = 3e-5, drop_frac=0.2,
                                        bidirectional=True, l1_norm = 1e-4, batch_norm = [False, False])

seed = 42
np.random.seed(seed)

scratch_path = '/global/scratch/jagar/'

run_id = sf.check_folder_exist(scratch_path + run_id)

sf.Make_folder(run_id)

model_name = run_id + 'start'
keras.models.save_model(model, run_id + '/start_seed_{0:03d}.h5'.format(seed))

#tbCallBack = keras.callbacks.TensorBoard(
#    log_dir= run_id + '/', histogram_freq=0, write_graph=True, write_images=True)

filepath = run_id + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=True, mode='min', period=1)

logger = keras.callbacks.CSVLogger(run_id + '/log.csv', separator=',', append=True)

history = model.fit(np.atleast_3d(Resonance), np.atleast_3d(Resonance), epochs=25000,
          batch_size=1800, validation_data=(np.atleast_3d(Resonance), np.atleast_3d(Resonance)),
          callbacks=[checkpoint, logger])
