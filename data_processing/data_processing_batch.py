from array import array
import soundfile as sf
import numpy as np
from scipy import signal
from pydub import AudioSegment
import glob
import os
import random
import librosa
import auditok
import matplotlib.pyplot as plt

def float2pcm(sig, dtype='int16'):
	#https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
	"""Convert floating point signal with a range from -1 to 1 to PCM.
	Any signal values outside the interval [-1.0, 1.0) are clipped.
	No dithering is used.
	Note that there are different possibilities for scaling floating
	point numbers to PCM numbers, this function implements just one of
	them.  For an overview of alternatives see
	http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
	Parameters
	----------
	sig : array_like
		Input array, must have floating point type.
	dtype : data type, optional
		Desired (integer) data type.
	Returns
	-------
	numpy.ndarray
		Integer data, scaled and clipped to the range of the given
		*dtype*.
	See Also
	--------
	pcm2float, dtype
	"""
	# sig = np.asarray(sig)
	if sig.dtype.kind != 'f':
		raise TypeError("'sig' must be a float array")
	dtype = np.dtype(dtype)
	if dtype.kind not in 'iu':
		raise TypeError("'dtype' must be an integer type")

	i = np.iinfo(dtype)
	abs_max = 2 ** (i.bits - 1)
	offset = i.min + abs_max
	return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def speech2total_ratio(np_array, sr = 16000, sw = 2, ch = 1, energy_thresh = 50):
	#loading data to auditok
	audio_bytes = float2pcm(np_array, dtype="int16").tobytes()
	main_region = auditok.load(audio_bytes, sr=sr, sw=2, ch=1)
	split_regions = auditok.split(main_region,
	min_dur=0.05,     # minimum duration of a valid audio event in seconds, default 0.2
		max_dur=4,       # maximum duration of an event, default 4
		max_silence=0.3, # maximum duration of tolerated continuous silence within an event, default 0.3
		energy_threshold=50 # threshold of detection, default 50
	) #https://auditok.readthedocs.io/en/latest/core.html
	
	total_speech_dur = 0 
	total_dur=main_region.duration
	# print(f"total_dur: {total_dur}s")
	for i, r in enumerate(split_regions):
		start = r.meta.start
		end = r.meta.end
		total_speech_dur += (end-start)
		# Regions returned by `split` have 'start' and 'end' metadata fields
		# print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
	ratio = total_speech_dur/total_dur
	return ratio

def get_statistics(final_lst_np, sr = 8000, energy_thresh=50, save_figname = None):
	# ratio_lst = []
	all_cleans = final_lst_np[:, :, 0]
	ratio_lst = [speech2total_ratio(clean, sr, energy_thresh= energy_thresh) for clean in all_cleans]
	plt.hist(ratio_lst, density=True, bins=10, range = (0., 1.))  # density=False would make counts
	plt.xlabel("speech2total_ratio")
	plt.ylabel("amount_normed")
	plt.title(f"total_data_distribution~{len(all_cleans)} chunks)")


	if save_figname is not None:
		plt.savefig(save_figname)

def select_data(input_total_data,ratio_threshold = (0.8, 1.0), total_amount = 10, sr = 8000, energy_thresh = 50):
	print("selecting data based on s2t_ratio, total amount...")

	#total amount in seconds
	total_num = 0
	training_lst = [] 
	val_lst = []
	for clean_noisy in input_total_data:
		clean = clean_noisy[:, 0]
		ratio = speech2total_ratio(clean, sr = sr, energy_thresh = energy_thresh)
		if ratio_threshold[0]<=ratio<=ratio_threshold[1] and total_num<total_amount:
			# print("ratio",ratio)
			training_lst.append(clean_noisy[np.newaxis, ...])
			total_num+=1

		else :
			val_lst.append(clean_noisy[np.newaxis, ...])
	train_npy =	np.concatenate(training_lst, axis = 0) #pending change! for each second: check ratio, def select_data(ratio_threshold, total_amount)
	val_npy =	np.concatenate(val_lst, axis = 0) #pending change! for each second: check ratio, def select_data(ratio_threshold, total_amount)

	return train_npy, val_npy


def time_align(audio_clean, audio_noisy):
	#question! Should I pad zero or 1e-12

	#correlation example
	#https://www.youtube.com/watch?v=RO8s1TrElEw&ab_channel=DavidDorran
	#what does same mode do
	#https://stackoverflow.com/questions/20036663/understanding-numpys-convolve
	clean = audio_clean.sample_array
	noisy = audio_noisy.sample_array
	#pad till same length
	if len(clean)!=len(noisy):
		zeros_array = np.zeros(( abs(len(clean)-len(noisy)), ), dtype = np.float32)
		print("initial clean, y2", clean, noisy)
		if(len(clean) < len(noisy)):
			clean = np.append(clean, zeros_array)
			print("padded clean", clean)
		elif(len(clean) > len(noisy)):
			noisy = np.append(noisy, zeros_array)
			print("padded noisy", noisy)

	#calculate shift
	corr = signal.correlate(clean, noisy, mode='full')
	lags_array = signal.correlation_lags(len(clean),len(noisy), mode="full")
	lag = lags_array[np.argmax(corr)] #noisy should shift lag to match y1

	#shift the array, return aligned and equal length array
	if lag>0:
		# print(f"clean{clean}, noisy{noisy}, noisy should shift right {lag} samples to match clean")
		noisy_padded = np.pad(noisy, (lag, 0), 'constant', constant_values=0)#shift right, pad zero left
		noisy_shifted =noisy_padded[:len(noisy)] 
	elif lag<0:
		# print(f"clean{clean}, noisy{noisy}, noisy should shift left {abs(lag)} samples to match clean")
		noisy_padded = np.pad(noisy, (0, abs(lag)), 'constant', constant_values=0)#shift left, pad zero right
		noisy_shifted = noisy_padded[abs(lag):]
	else:
		noisy_shifted = noisy

	audio_clean.sample_array, audio_noisy.sample_array = clean, noisy_shifted
	# return clean, noisy_shifted
	return audio_clean, audio_noisy

def normalize_audio_volumes(audio_clean, audio_noisy):
	#adjust noisy only
	change_in_db = audio_clean.dbfs-audio_noisy.dbfs
	change_in_float = 10 ** (change_in_db / 20) #10 or 20?
	audio_noisy.sample_array = audio_noisy.sample_array*change_in_float
	# print(audio_clean.dbfs, audio_noisy.dbfs, change_in_db, change_in_float)
	# audio_noisy_one_hot = AudioSegment.from_wav(audio_noisy.audio_file_path) 
	# audio_noisy_one_hot_volume_adjusted = audio_noisy_one_hot.apply_gain(audio_clean.dbfs - audio_noisy.dbfs)

	# audio_noisy_volume_adjusted = np.array(audio_noisy_one_hot_volume_adjusted.get_array_of_samples()).astype(np.float32) / \
	# 	(2**(audio_noisy.bit_rate - 1))
	
	return audio_clean, audio_noisy
def test_rms(audio_clean, audio_noisy):
	audio_clean_array, audio_noisy_array = audio_clean.sample_array, audio_noisy.sample_array
	rms_audio_clean = sum(x**2 for x in audio_clean_array)/len(audio_clean_array)
	rms_audio_noisy = sum(x**2 for x in audio_noisy_array)/len(audio_noisy_array)
	print("heu",rms_audio_clean, rms_audio_noisy)
class audio_reader():
	def __init__(self, audio_file_path = None, if_pad_end = False, if_noise_floor = False, sample_rate = None, np_array = None):
		self.audio_file_path = audio_file_path
		if np_array is not None: 
			try:
				self.sample_array = np.load(np_array)
			except:
				self.sample_array = np_array
			self.sample_rate = 8000
			self.dbfs = self.get_loudness()
		else:
			if sample_rate is not None and sample_rate!=8000:
				self.sample_array, self.sample_rate = librosa.load(audio_file_path, sample_rate)
			else:
				self.sample_array, self.sample_rate = sf.read(audio_file_path) #pengding change
			self.dbfs = self.get_loudness()

		self.len = len(self.sample_array)

		if if_pad_end:
			num_pad = self.sample_rate - self.len%self.sample_rate 
			self.sample_array = np.pad(self.sample_array, (0, num_pad), constant_values=0)	
		if if_noise_floor:
			self.sample_array[np.where(self.sample_array==0)]=1e-12

	def get_bit_rate(self):
		aud = sf.SoundFile(self.audio_file_path)
		if self.subtype=="FLOAT":
			bit_rate = 32 #32 bit float audio
		else:
			bit_rate = int(aud.subtype.split("_")[-1])
		return bit_rate
	def get_sample_rate(self):
		aud = sf.SoundFile(self.audio_file_path)
		sample_rate = int(aud.samplerate)
		return sample_rate
	def get_loudness(self):
		aud = AudioSegment.from_wav(self.audio_file_path)
		return aud.dBFS
	def write(self, name):
		sf.write(name, self.sample_array, self.sample_rate) 



def process_one_pair(clean_path, noisy_path, sr):

	noisy_audio = audio_reader(audio_file_path = noisy_path, sample_rate = sr)
	clean_audio = audio_reader(audio_file_path = clean_path, sample_rate = sr)

	#normalize volume
	clean_audio_normed, noisy_audio_normed = normalize_audio_volumes(clean_audio, noisy_audio)

	#shift audio
	clean_audio_normed_aligned, noisy_audio_normed_aligned = time_align(clean_audio_normed, noisy_audio_normed)

	#convert to diff chunks EACH 1s
	num_chunks = clean_audio_normed_aligned.sample_array.shape[0]//clean_audio_normed_aligned.sample_rate
	if num_chunks==0:
		return None
	array_clean = clean_audio_normed_aligned.sample_array[:num_chunks*clean_audio_normed_aligned.sample_rate]
	array_noisy = noisy_audio_normed_aligned.sample_array[:num_chunks*noisy_audio_normed_aligned.sample_rate]

	#noise floor
	array_clean[np.where(array_clean==0)]=1e-12
	array_noisy[np.where(array_noisy==0)]=1e-12

	#reshape
	array_clean = np.reshape(array_clean, (num_chunks, -1))[..., np.newaxis]
	array_noisy = np.reshape(array_noisy, (num_chunks, -1))[..., np.newaxis]

	final_file = np.concatenate((array_clean, array_noisy), axis = -1)

	return final_file

def process_dirs_of_files(parent_dir , sr):
	final_lst = []
	clean_paths = sorted(glob.glob(f"{parent_dir}/clean/*.wav"))
	noisy_paths = sorted(glob.glob(f"{parent_dir}/noisy/*.wav"))

	for clean_path, noisy_path in zip(clean_paths, noisy_paths):
		assert clean_path.split("/")[-1].split("_")[0] == noisy_path.split("/")[-1].split("_")[0]
		output = process_one_pair(clean_path, noisy_path, sr)
		if output is not None:
			final_lst.append(output)
	return final_lst
if __name__ == '__main__':
	
	SEED = 265
	sr = 8000
	random.seed(SEED)
	if_get_status = False
	ratio_threshold = (0.8, 1.0)
	energy_thresh = 50
	total_train_amount = 10
	parent_dir = "RATs_data_mid"
	
	#convert everything into 1s chunks
	final_lst = process_dirs_of_files(parent_dir, sr = sr)
	random.shuffle(final_lst)
	final_lst_np = np.concatenate(final_lst, axis = 0)
	
	#get ratio histogram
	if if_get_status:
		total_statistics =  get_statistics(final_lst_np, sr = sr,energy_thresh = energy_thresh, save_figname = "total_statistics.png")
	
	#select data based on amount & ratio threshold
	training_npy, val_npy = select_data(input_total_data = final_lst_np,ratio_threshold = ratio_threshold,  total_amount =total_train_amount, sr = sr, energy_thresh= energy_thresh)

	np.save(f"{parent_dir}/rats_small_train_thd{ratio_threshold[0]},{ratio_threshold[1]}_sr{sr}_len{len(training_npy)}.npy", training_npy)
	np.save(f"{parent_dir}/rats_small_valid_thd{ratio_threshold[0]},{ratio_threshold[1]}_sr{sr}_len{len(val_npy)}.npy", val_npy)

	