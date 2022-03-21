import tensorflow as tf
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[2], True)
device = "1"
import os
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=device



devices = tf.config.list_physical_devices('GPU')
try:
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # tf.config.experimental.set_memory_growth(devices[1], True)
    tf.config.experimental.set_memory_growth(devices[int(device)], True)
    # tf.config.experimental.set_memory_growth(devices[3], True)

    print("Success in setting memory growth")
except:
    print("Failed to set memory growth, invalid device or cannot modify virtual devices once initialized.")


from datetime import datetime
# from model import  distortion, compressor, compressor_smoothing_gain_cell,data_loader, data_loader_test, , signal_chain_gpu_from_numpy
from model_freq_domain import signal_chain_gpu,SpectralLoss
import numpy as np
import math
import soundfile as sf
import time
import yaml

def loss_function(tar_real, predictions):
	loss = SpectralLoss()
	return loss(tar_real, predictions)
def save_audio(pred, sr, audio_name):
	pred =tf.reshape(pred, (1, -1))
	pred = pred.numpy()[0].astype(np.float32)
	sf.write(audio_name, pred, sr,subtype='FLOAT')

if __name__=="__main__":
	#config

	with open ('config/training.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	with open ('config/model.yaml', 'r') as f:
		cfg_model = yaml.safe_load(f)

	#make data pending change make valid data!
	clean_noisy_tuple_train = np.load(cfg['data_path_train'])
	clean_noisy_tuple_valid = np.load(cfg['data_path_valid'])

	clean_audio_train = clean_noisy_tuple_train[:, :,0]
	noisy_audio_train = clean_noisy_tuple_train[:, :,1] #(10, 45056) #batch, len

	clean_audio_valid = clean_noisy_tuple_valid[:, :,0]
	noisy_audio_valid = clean_noisy_tuple_valid[:, :,1] #(10, 45056) #batch, len

	ds_clean_noisy_train = tf.data.Dataset.from_tensor_slices((clean_audio_train, noisy_audio_train)).batch(cfg['batch_size']) #(no_of_chunks, sample_Array_len)
	ds_clean_noisy_valid = tf.data.Dataset.from_tensor_slices((clean_audio_valid, noisy_audio_valid)).batch(cfg['batch_size']) #(no_of_chunks, sample_Array_len)
	print("data loaded")
	#define model
	model = signal_chain_gpu(EQ_cfg = cfg_model['FIRfilter'], DRC_cfg = cfg_model['compressor'], waveshaper_cfg=cfg_model['distortion'], noise_cfg = cfg_model['filtered_noise'])
	optimizer = tf.keras.optimizers.Adam(cfg['learning_rate'], beta_1=0.9, beta_2=0.98,
										epsilon=1e-9)
	print("optim created")

	@tf.function
	def train_step(clean_audio, noisy_audio):
		with tf.GradientTape() as tape:
			predictions = model(clean_audio)
			total_loss = loss_function(noisy_audio, predictions)
		batch_loss_train(total_loss)

		gradients = tape.gradient(total_loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		return predictions, total_loss


	@tf.function
	def val_step(clean_audio, noisy_audio):
		predictions = model(clean_audio)
		total_loss = loss_function(noisy_audio, predictions)
		batch_loss_valid(total_loss)

		return predictions, total_loss

	#checkpoint path
	date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

	checkpoint_path = os.path.join("checkpoints", date_time+"_"+cfg['data_path_train'].split("/")[-1][:-4])
	batch_loss_train = tf.keras.metrics.Mean(name='batch_loss_train')
	batch_loss_valid = tf.keras.metrics.Mean(name='batch_loss_valid')

	train_log_dir = checkpoint_path + '/train'
	val_log_dir = checkpoint_path + '/val'





	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	valid_summary_writer = tf.summary.create_file_writer(val_log_dir)
	ckpt = tf.train.Checkpoint(model = model,
							optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=cfg['max_to_keep'])
	# if a checkpoint exists, restore the latest checkpoint. check how to do this!
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		last_end_epoch = int(ckpt_manager.latest_checkpoint.split("/")[-1].split("-")[-1])
		print('Latest checkpoint restored!!',ckpt_manager.latest_checkpoint)
	else:
		last_end_epoch = 0


	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	with open(checkpoint_path+"/training.yaml", 'w') as f:
		yaml.dump(cfg, f)

	with open(checkpoint_path+"/model.yaml", 'w') as f:
		yaml.dump(cfg_model, f)
	
	prev_loss = 999
	print("start training")

	for epch in range (cfg['epochs']):
		batch_loss_train.reset_states()
		batch_loss_valid.reset_states()

		# for i, (clean_audio, noisy_audio) in enumerate(ds_clean_noisy_train.take(len(clean_noisy_tuple_train))):
		for i, (clean_audio, noisy_audio) in enumerate(ds_clean_noisy_train):
			# print(f"inspect shape", clean_audio.shape, noisy_audio.shape)
			last_time = time.time()
			pred_train, loss = train_step(clean_audio, noisy_audio) #pred should be 16 bit also 16-bit PCM -32768 +32767 int16
			total_secs = round(time.time() - last_time)
			print(f"epoch {epch},step {i}, loss:{batch_loss_train.result()}, total secs:{total_secs} ")
		
		with train_summary_writer.as_default():
			tf.summary.scalar('loss_train', batch_loss_train.result(), step=epch)
		
		# for i, (clean_audio, noisy_audio) in enumerate(ds_clean_noisy_valid.take(len(clean_noisy_tuple_valid))):
		for i, (clean_audio, noisy_audio) in enumerate(ds_clean_noisy_valid):

			last_time = time.time()
			_, loss = val_step(clean_audio, noisy_audio) #pred should be 16 bit also 16-bit PCM -32768 +32767 int16
			total_secs = round(time.time() - last_time)
			
			print(f"valid epoch {epch},step {i}, loss:{batch_loss_valid.result()}, total secs:{total_secs} ")
		with valid_summary_writer.as_default():
			tf.summary.scalar('loss_valid', batch_loss_valid.result(), step=epch)	
		
		if batch_loss_valid.result() < prev_loss or epch ==cfg['epochs']-1 :
			save_audio(pred_train, cfg['sr'],audio_name= f"{checkpoint_path}/ep{epch}{loss:.2f}.wav")
			prev_loss = batch_loss_valid.result()
		ckpt_save_path = ckpt_manager.save()
