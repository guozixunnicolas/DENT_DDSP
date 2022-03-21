from .utils import *
from .core import exp_sigmoid, frequency_filter,compressor_time_averaged, distortion_actan, clip_by_value

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import math
import numpy as np
from typing import Any, Dict, Optional, Sequence, Text, TypeVar
import soundfile as sf
from scipy.io.wavfile import write
import yaml
# import ddsp
TensorDict = Dict[Text, tf.Tensor]



class SpectralLoss(tf.keras.layers.Layer):

	def __init__(self,
				fft_sizes=(2048, 1024, 512, 256, 128, 64),
				loss_type='L1',
				mag_weight=1.0,
				logmag_weight=0.0,
				name='spectral_loss'):

		super(SpectralLoss,self).__init__()
		self.fft_sizes = fft_sizes
		self.loss_type = loss_type
		self.mag_weight = mag_weight
		self.logmag_weight = logmag_weight

	def safe_log(self,x, eps=1e-5):
		safe_x = tf.where(x <= eps, eps, x)
		return tf.math.log(safe_x)

	def tf_float32(self, x):
		if isinstance(x, tf.Tensor):
			return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
		else:
			return tf.convert_to_tensor(x, tf.float32)

	def stft(self, audio, frame_size=2048, overlap=0.75, pad_end=True):
		assert frame_size * overlap % 2.0 == 0.0

		# Remove channel dim if present.
		audio = self.tf_float32(audio)
		if len(audio.shape) == 3:
			audio = tf.squeeze(audio, axis=-1)

		s = tf.signal.stft(
			signals=audio,
			frame_length=int(frame_size),
			frame_step=int(frame_size * (1.0 - overlap)),
			fft_length=int(frame_size),
			pad_end=pad_end)
		return s
	
	def mean_difference(self, target, value, loss_type='L1'):

		difference = target - value
		loss_type = loss_type.upper()
		if loss_type == 'L1':
			return tf.reduce_mean(tf.abs(difference))

	def call(self, target_audio, audio):
		loss = 0.0
		# Compute loss for each fft size.
		for fft_size in self.fft_sizes:
			
			target_mag = self.stft(target_audio, frame_size=fft_size, overlap=0.75, pad_end=True)
			value_mag = self.stft(audio, frame_size=fft_size, overlap=0.75, pad_end=True)
			target_mag = tf.cast(tf.abs(target_mag), tf.float32)
			value_mag = tf.cast(tf.abs(value_mag), tf.float32)
			# Add magnitude loss.
			if self.mag_weight > 0:
				loss += self.mag_weight * self.mean_difference(
					target_mag, value_mag, self.loss_type)
			# Add logmagnitude loss, reusing spectrogram.
			if self.logmag_weight > 0:
				target = self.safe_log(target_mag)
				value = self.safe_log(value_mag)
				loss += self.logmag_weight * self.mean_difference(
					target, value, self.loss_type)
		return loss


class Processor(tf.keras.layers.Layer):
	"""Abstract base class for signal processors.
	Since most effects / synths require specificly formatted control signals
	(such as amplitudes and frequenices), each processor implements a
	get_controls(inputs) method, where inputs are a variable number of tensor
	arguments that are typically neural network outputs. Check each child class
	for the class-specific arguments it expects. This gives a dictionary of
	controls that can then be passed to get_signal(controls). The
	get_outputs(inputs) method calls both in succession and returns a nested
	output dictionary with all controls and signals.
	"""

	def __init__(self, name: Text, trainable: bool = True):
		super().__init__(name=name, trainable=trainable, autocast=False)

	def call(self,
					*args: tf.Tensor,
					return_outputs_dict: bool = False,
					**kwargs) -> tf.Tensor:
		"""Convert input tensors arguments into a signal tensor."""
		# Don't use `training` or `mask` arguments from keras.Layer.
		for k in ['training', 'mask']:
			if k in kwargs:
				_ = kwargs.pop(k)

		controls = self.get_controls(*args, **kwargs)
		# print("wtf!!", controls['threshold_dist'].trainable)
		signal = self.get_signal(**controls)
		if return_outputs_dict:
			return dict(signal=signal, controls=controls)
		else:
			return signal

	def get_controls(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> TensorDict:
		"""Convert input tensor arguments into a dict of processor controls."""
		raise NotImplementedError

	def get_signal(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> tf.Tensor:
		"""Convert control tensors into a signal tensor."""
		raise NotImplementedError


#distortion
class distortion(Processor):
	def __init__(self, distortion_threshold_init, name='distortion'):
		super().__init__(name=name)
		self.distortion_threshold_init = distortion_threshold_init

	def build(self, input_shape):  # Create the state of the layer (weights)
		self.distortion_threshold = tf.Variable(name="distortion_threshold", initial_value=self.distortion_threshold_init, trainable=True)	

	def get_controls(self, audio):
		return {"audio":audio, "threshold_dist":self.distortion_threshold}

	def get_signal(self, audio, threshold_dist):
		out = distortion_actan(audio, threshold_dist)
		# print("omg", self.distortion_threshold.trainable)

		return out

	# def call(self,input_file):
	# 	distorted=(2/math.pi) * tf.math.atan( (math.pi/self.distortion_threshold) * input_file)
	# 	return distorted

#FIR
class FIRFilter(Processor):
	"""Linear time-varying finite impulse response (LTV-FIR) filter."""

	def __init__(self,
							 window_size=257,
							#  scale_fn=exp_sigmoid,
							scale_fn = clip_by_value,
							 name='fir_filter',
							 magnitude_init = None,
							 n_frequency_bins = None,
							 n_frames = None ):
		super().__init__(name=name)
		self.window_size = window_size
		self.scale_fn = scale_fn
		self.magnitude_init = magnitude_init
		self.n_frequency_bins = n_frequency_bins
		self.n_frames = n_frames
	def build(self, input_shape):  
		if self.magnitude_init is None:
			initializer = tf.random_normal_initializer() #pending change!
			if self.n_frames is None: 
				# magnitudes_init = initializer(shape=(input_shape[0],self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies
				magnitudes_init = initializer(shape=(1,self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies

			else:
				# magnitudes_init = initializer(shape=(input_shape[0],self.n_frames, self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies
				magnitudes_init = initializer(shape=(1,self.n_frames, self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies

		else:
			magnitudes_init = self.magnitude_init

		self.magnitudes= tf.Variable(name = "fir_mag",
			initial_value=magnitudes_init,
			trainable=True)


	def get_controls(self, audio ):
		"""Convert network outputs into magnitudes response.
		Args:
			audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
			magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
				n_filter_banks].
		Returns:
			controls: Dictionary of tensors of synthesizer controls.
		"""
		# Scale the magnitudes.
		if self.scale_fn is not None:
			magnitudes = self.scale_fn(self.magnitudes)
		else:
			magnitudes = self.magnitudes
		return  {'audio': audio, 'magnitudes': magnitudes}

	def get_signal(self, audio, magnitudes):
		"""Filter audio with LTV-FIR filter.
		Args:
			audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
			magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
				Expects float32 that is strictly positive.
		Returns:
			signal: Filtered audio of shape [batch, n_samples, 1].
		"""
		
		out = frequency_filter(audio,
								magnitudes,
								window_size=self.window_size)
		# print(out.shape,out, "inside")
		return out

#DRC
class DynamicRangeCompressor(Processor):
	def __init__(self, sr,threshold_init,ratio_init, makeup_init,attack_init, release_init, downsample_factor = 16.0, name = "drc"):
		super().__init__(name=name)
		self.threshold_init = threshold_init
		self.ratio_init = ratio_init
		self.makeup_init = makeup_init
		# self.attack_time_init= attack_init
		# self.release_time_init = release_init
		self.attack_time_init=tf.math.exp(-log10(9.0)/(sr*attack_init*1.0E-3*downsample_factor))
		self.release_time_init=tf.math.exp(-log10(9.0)/(sr*release_init*1.0E-3*downsample_factor))
		self.sr = sr
		self.downsample_factor = downsample_factor
	def build(self, input_shape): 
		self.threshold = tf.Variable(name="threshold", initial_value=self.threshold_init, trainable=True)
		self.ratio = tf.Variable(name="ratio", initial_value=self.ratio_init, trainable=True)
		self.makeup = tf.Variable(name="makeup", initial_value=self.makeup_init, trainable=True)
		self.attack_time = tf.Variable(name="attack_time_constant", initial_value=self.attack_time_init, trainable=True) #pendin change
		self.release_time = tf.Variable(name="release_time_constant", initial_value=self.release_time_init, trainable=True) #pendin change

	def get_controls(self, audio):
		
		return  {'audio': audio, 'sr':self.sr,'threshold': self.threshold, 'ratio':self.ratio, 'makeup':self.makeup,'attack_time':self.attack_time, 'release_time':self.release_time, 'downsample_factor':self.downsample_factor}
	def get_signal(self, audio, sr, threshold, ratio,makeup, attack_time,  release_time, downsample_factor):
		out = compressor_time_averaged(audio,sr,threshold, ratio,makeup, attack_time,  release_time, downsample_factor)
		return out

#Filtered noise
class FilteredNoise(Processor):
	def __init__(self,
				# n_samples=64000,
				window_size=257,
				# scale_fn=exp_sigmoid,
				scale_fn = clip_by_value,
				name='filtered_noise',
				initial_bias_init=None,
				magnitude_init = None,
				n_frequency_bins = None,
				n_frames = None 
				):
		super().__init__(name=name)
		# self.n_samples = n_samples
		self.window_size = window_size
		self.scale_fn = scale_fn
		self.initial_bias_init = initial_bias_init
		self.magnitude_init = magnitude_init
		self.n_frequency_bins = n_frequency_bins
		self.n_frames = n_frames
	def build(self, input_shape):  # Create the state of the layer (weights)

		if self.magnitude_init is None:
			initializer = tf.random_normal_initializer() #pending change!
			if self.n_frames is None: 
				magnitude_init = initializer(shape=(1,self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies
			else:
				magnitude_init = initializer(shape=(1,self.n_frames, self.n_frequency_bins), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies
		else:	
			magnitude_init = self.magnitude_init

		self.magnitudes= tf.Variable(name = "noise_mag",
			initial_value=magnitude_init,
			trainable=True)
		
		if self.initial_bias_init is None:
			initial_bias_init = initializer(shape=(1, ), dtype='float32') #batch_size, n_frames, n_frequencies or batch_size, n_frames, n_frequencies
			self.initial_bias= tf.Variable(name = "noise_amp_bias",
				initial_value=initial_bias_init,
				trainable=True)
		else:
			self.initial_bias= tf.Variable(name = "noise_amp_bias",
				initial_value=self.initial_bias_init,
				trainable=True)
		# print("hello")
	def get_controls(self, audio):
		"""Convert network outputs into a dictionary of synthesizer controls.
		Args:
		magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
			n_filter_banks].
		Returns:
		controls: Dictionary of tensors of synthesizer controls.
		"""
		# Scale the magnitudes.
		if self.scale_fn is not None:
			magnitudes = self.scale_fn(self.magnitudes + self.initial_bias)
		else:
			# magnitudes = self.magnitudes #before no amp control
			magnitudes = self.magnitudes + self.initial_bias
		noise_sig = tf.random.uniform(
			[magnitudes.shape[0], audio.shape[-1]], minval=-1.0, maxval=1.0)
		return {'noise_sig':noise_sig, 'magnitudes': magnitudes}

	def get_signal(self, noise_sig,magnitudes):
		"""Synthesize audio with filtered white noise.
		Args:
		magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
			Expects float32 that is strictly positive.
		Returns:
		signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
		"""
		out = frequency_filter(noise_sig,
									magnitudes,
									window_size=self.window_size)
		return out



class signal_chain_gpu(tf.keras.Model):
	def __init__(self, EQ_cfg = None, DRC_cfg = None, waveshaper_cfg=None, noise_cfg = None):
		super(signal_chain_gpu, self).__init__()

		#processor_dict: {proc1: parameter1, proc2:parameter2}
		# self.proc_lst = []

		if waveshaper_cfg is not None:
			self.waveshaper = distortion(**waveshaper_cfg)
			# self.proc_lst.append(self.waveshaper)	
		if DRC_cfg is not None:
			self.DRC = DynamicRangeCompressor(**DRC_cfg)
			# self.proc_lst.append(self.DRC)
		
		if EQ_cfg is not None:
			self.EQ = FIRFilter(**EQ_cfg)
			# self.proc_lst.append(self.EQ)

		if noise_cfg is not None:
			self.noise = FilteredNoise(**noise_cfg)
		else:
			self.noise = None
			# self.proc_lst.append(self.noise)
		
	def call(self, output):

		if self.waveshaper is not None:
			output = self.waveshaper(output)

		if self.DRC is not None:
			output = self.DRC(output)

		if self.EQ is not None:

			output = self.EQ(output)	

		if self.noise is not None:
			output = output+self.noise(output)

		return output


if __name__ == "__main__":

	with open ('../config/model.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	sample_rate = 16000
	frame_rate = 100
	
	# audio_path = "../440_1_freq.wav"
	audio_path = "../input_clean_32FP.wav"
	audio,sample_rate = sf.read(audio_path)
	# n_seconds = audio.size // sample_rate
	n_seconds = 10.0
	n_frames = int(n_seconds * frame_rate)
	n_samples = int(n_seconds * sample_rate)

	audio = audio[np.newaxis, :]
	audio_trimmed = audio[:, :n_samples]

	"""specplot(audio_trimmed[0], fig_name="sinori.png")



	magnitudes = create_gauss_mag(n_seconds, sample_rate, frame_rate, cfg['n_frequencies']) #[batch, n_frames, n_freq]

	# fir_filter = FIRFilter(scale_fn=None, magnitude_init = magnitudes)
	# fir_filter = FIRFilter(scale_fn=None, n_frequency_bins=  cfg['n_frequencies'])
	fir_filter = FIRFilter(**cfg['FIRfilter'])

	audio_out = fir_filter(audio_trimmed)
	print(fir_filter.trainable_variables)
	sf.write("filtered_audio.wav", audio_out[0].numpy(), sample_rate) 

	#test filtered noise
	# cfg['filtered_noise']['magnitude_init'] = tf.cast(magnitudes, tf.float32) #change here to pass in magn
	noise_gen = FilteredNoise(**cfg['filtered_noise'])
	# noise_gen = FilteredNoise(scale_fn=None, initial_bias_init= -5.0,magnitude_init= tf.cast(magnitudes, tf.float32))
	# noise_gen = FilteredNoise(initial_bias_init= 0.0,n_frequency_bins=  cfg['n_frequencies'])
	# noise, raw_noise = noise_gen(4*sample_rate)
	noise = noise_gen(audio_trimmed)

	sf.write("noise.wav",noise[0].numpy(), sample_rate) 

	spec_gainplot(noise_gen.magnitudes.numpy(), "noisegain.png")
	specplot(noise[0].numpy(), fig_name="noise.png")

	#test compressor
	comp = DynamicRangeCompressor(**cfg['compressor'])

	audio_comped = comp(audio_trimmed)
	sf.write("comped_sin.wav", audio_comped[0].numpy(), sample_rate)

	#test dist
	dist = distortion(**cfg['distortion'])
	audio_dist = dist(audio_trimmed)
	sf.write("disted_sin.wav", audio_dist[0].numpy(), sample_rate) 
	print("check ", dist.distortion_threshold.trainable,dist.trainable_variables, dist.trainable_weights)"""

	#test processor group
	signal_chain_gpu = signal_chain_gpu(EQ_cfg = cfg['FIRfilter'], DRC_cfg = cfg['compressor'], waveshaper_cfg=cfg['distortion'], noise_cfg = cfg['filtered_noise'])
	# @tf.function
	def train_step(inp):
		out = signal_chain_gpu(inp)
		return out
	audio_out = train_step(audio_trimmed)
	# print(f"{signal_chain_gpu.trainable_variables}")
	# audio_out = signal_chain_gpu(audio_trimmed)
	sf.write("final_out.wav", audio_out[0].numpy(), sample_rate) 

	# print("audio out", audio_out.shape)
	# dense = SimpleDense()
	# a = tf.convert_to_tensor([[1., 2., 3., 3.]])
	# b = tf.convert_to_tensor([[1., 2., 3., 3., 5.]])
	# print(dense, dense((a, b)), dense.trainable_variables)