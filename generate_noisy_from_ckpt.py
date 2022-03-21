import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# 	try:
# 		# Currently, memory growth needs to be the same across GPUs
# 		for gpu in gpus:
# 			tf.config.experimental.set_memory_growth(gpu, True)
# 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 	except RuntimeError as e:
# 		# Memory growth must be set before GPUs have been initialized
# 		print(e)
# device = "2"


import os
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=device
from datetime import datetime
from model_freq_domain import signal_chain_gpu,SpectralLoss
import numpy as np
import math
import soundfile as sf
import time
# from LSD import audio_reader
import yaml
from data_processing.data_processing_batch import audio_reader
import glob
from tqdm import tqdm
def save_audio(pred, sr, audio_name):
	pred =tf.reshape(pred, (1, -1))
	pred = pred.numpy()[0].astype(np.float32)
	sf.write(audio_name, pred, sr,subtype='PCM_16')


if __name__=="__main__":

	#config

	ckpt_path = "checkpoints/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10"
	step = "ckpt-132"

	with open (ckpt_path+'/training.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	with open (ckpt_path+'/model.yaml', 'r') as f:
		cfg_model = yaml.safe_load(f)

	noise_adjustment = 4.0
	data_dir = "dump_clean/raw/org/data_src_valid_8k/data/"
	# new_dir = "results_folder/"+ckpt_path.split("/")[-1]+"/"+step
	new_dir = "results_folder/"+ckpt_path.split("/")[-1]+"/"+step +f"-{str(noise_adjustment)}"

	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	#define model
	model = signal_chain_gpu(EQ_cfg = cfg_model['FIRfilter'], DRC_cfg = cfg_model['compressor'], waveshaper_cfg=cfg_model['distortion'], noise_cfg = cfg_model['filtered_noise'])
	optimizer = tf.keras.optimizers.Adam(cfg['learning_rate'], beta_1=0.9, beta_2=0.98,
										epsilon=1e-9)

	ckpt = tf.train.Checkpoint(model=model)
	ckpt.restore(ckpt_path+"/"+step)

	#make data
	# clean_paths = [x for x in glob.glob(data_dir+"/*.wav")]
	clean_paths = [x for x in glob.glob(data_dir+"/**/*.wav")]

	# @tf.function
	# def forward(inp_audio):
	# 	return model(inp_audio)
	@tf.function
	def forward(inp_audio, noise_adjustment = None):
		# if self.waveshaper is not None:
		output = model.waveshaper(inp_audio)
		output = model.DRC(output)
		output = model.EQ(output)	
		noise = model.noise(output)
		if noise_adjustment is not None:
			print(f"noise adjust:{noise_adjustment}")
			adj_linear = 20**(noise_adjustment/20)
			noise*=adj_linear

		return output+noise


	for i, clean_path in tqdm(enumerate(clean_paths)):
		print("processing", clean_path)
		# audio=audio_reader(np_array = clean_path,if_pad_end = True, if_noise_floor = True, sample_rate = cfg["sampling_rate"])
		audio=audio_reader(audio_file_path = clean_path,if_pad_end = True, if_noise_floor = True, sample_rate = cfg["sampling_rate"])

		inp_audio = audio.sample_array[np.newaxis, :] #pending change 
		# out_name = new_dir+"/"+ clean_path.split("/")[-1].split(".")[0]+".wav"
		# out_name = new_dir+"/"+ clean_path.split("/")
		if not os.path.exists(new_dir+"/"+"/".join(clean_path.split("/")[-4:-1])):
			os.makedirs(new_dir+"/"+"/".join(clean_path.split("/")[-4:-1]))
		out_name = new_dir+"/"+"/".join(clean_path.split("/")[-4:])
		out = forward(inp_audio).numpy()

		save_audio(out[:, :audio.len], cfg["sampling_rate"],audio_name= out_name)
		print("Saved", out_name)


	##check number of param
	# import tensorflow.keras.backend as K
	# trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
	# non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])	

	# print('Total params: {:,}'.format(trainable_count + non_trainable_count))
	# print('Trainable params: {:,}'.format(trainable_count))
	# print('Non-trainable params: {:,}'.format(non_trainable_count))
	# print(model.waveshaper.distortion_threshold,\
	# 		model.DRC.threshold,model.DRC.ratio,model.DRC.makeup,model.DRC.attack_time, model.DRC.release_time,
	# 		model.EQ.magnitudes,\
	# 		model.noise.magnitudes, model.noise.initial_bias )