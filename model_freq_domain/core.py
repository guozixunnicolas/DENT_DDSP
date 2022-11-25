# from neural_noisy_speech.model.model import log10
import tensorflow as tf
import numpy as np
from typing import Any, Dict, Optional, Sequence, Text, TypeVar
from scipy import fftpack
import scipy
import soundfile as sf
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa
import math


def clip_by_value(x,max_value=0.5, threshold=1e-12):
	return tf.clip_by_value(x, clip_value_min = threshold, clip_value_max = max_value, name=None)




def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method: Text = 'linear',
             add_endpoint: bool = True) -> tf.Tensor:
  """Interpolates a tensor from n_frames to n_timesteps.
  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
      'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
      'window' uses overlapping windows (only for upsampling) which is smoother
      for amplitude envelopes with large frame sizes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].
  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
      'window'.
  """
  inputs = tf_float32(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
  elif method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs

def upsample_with_windows(inputs: tf.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> tf.Tensor:
	"""Upsample a series of frames using using overlapping hann windows.
	Good for amplitude envelopes.
	Args:
		inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
		n_timesteps: The time resolution of the output signal.
		add_endpoint: Hold the last timestep for an additional step as the endpoint.
		Then, n_timesteps is divided evenly into n_frames segments. If false, use
		the last timestep as the endpoint, producing (n_frames - 1) segments with
		each having a length of n_timesteps / (n_frames - 1).
	Returns:
		Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].
	Raises:
		ValueError: If input does not have 3 dimensions.
		ValueError: If attempting to use function for downsampling.
		ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
		true) or n_frames - 1 (if add_endpoint is false).
	"""
	inputs = tf_float32(inputs)

	if len(inputs.shape) != 3:
		raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
						'not {}.'.format(inputs.shape))

	# Mimic behavior of tf.image.resize.
	# For forward (not endpointed), hold value for last interval.
	if add_endpoint:
		inputs = tf.concat([inputs, inputs[:, -1:, :]], axis=1)

	n_frames = int(inputs.shape[1])
	n_intervals = (n_frames - 1)

	if n_frames >= n_timesteps:
		raise ValueError('Upsample with windows cannot be used for downsampling'
						'More input frames ({}) than output timesteps ({})'.format(
							n_frames, n_timesteps))

	if n_timesteps % n_intervals != 0.0:
		minus_one = '' if add_endpoint else ' - 1'
		raise ValueError(
			'For upsampling, the target the number of timesteps must be divisible '
			'by the number of input frames{}. (timesteps:{}, frames:{}, '
			'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
									add_endpoint))
	# Constant overlap-add, half overlapping windows.
	hop_size = n_timesteps // n_intervals
	window_length = 2 * hop_size
	window = tf.signal.hann_window(window_length)  # [window]
	# Transpose for overlap_and_add.
	x = tf.transpose(inputs, perm=[0, 2, 1])  # [batch_size, n_channels, n_frames]

	# Broadcast multiply.
	# Add dimension for windows [batch_size, n_channels, n_frames, window].
	x = x[:, :, :, tf.newaxis]
	window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
	x_windowed = (x * window)#batch, n_channel, n_frames, 1 * batch, n_channel, 1, window_size

	x = tf.signal.overlap_and_add(x_windowed, hop_size)

	# Transpose back.
	x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

	# Trim the rise and fall of the first and last window.
	return x[:, hop_size:-hop_size, :]

def downsample_with_windows(inputs, hparams):
	inputs = tf_float32(inputs) #batch, len
	framed_inputs = tf.signal.frame(inputs, hparams.win_length, hparams.hop_length, pad_end = False, axis = -1)
	# framed_inputs = tf.signal.frame(inputs, win_length, hop_length, pad_end = False, axis = -1) #batch, num_frame, win_length
	framed_inputs = framed_inputs[:, :, tf.newaxis, :] #batch, num_frame, 1, win_length
	window = tf.signal.hann_window(hparams.win_length)[tf.newaxis,tf.newaxis, :, tf.newaxis]  # [batch, num_frame, win_length, 1]
	framed_windowed_inputs = tf.matmul(framed_inputs, window)
	# framed_windowed_inputs = tf.einsum('bfow,bfwo', framed_inputs, window)
	out = framed_windowed_inputs[:, :, :,0]
	# print(f"inputs:{inputs.shape},frame input:{framed_inputs.shape}, window:{window.shape}, framed_windowed_inputs:{framed_windowed_inputs.shape}, out:{out.shape}")
	return out
def mag_phase_2_real_imag(mag, phase):     
	# print("type", type(mag), type) 
	# print("type",type(mag), type(phase) )
	cos_phase = tf.math.cos(phase)
	sin_phase = tf.math.sin(phase)
	r= tf.complex(mag*cos_phase, mag*sin_phase)
	return  r

def log10(x):
	numerator = tf.math.log(x)
	denominator = tf.math.log(10.0)
	return numerator / denominator
def tf_float32(x):
	"""Ensure array/tensor is a float32 tf.Tensor."""
	if isinstance(x, tf.Tensor):
		return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
	else:
		return tf.convert_to_tensor(x, tf.float32)

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
	"""Exponentiated Sigmoid pointwise nonlinearity.
	Bounds input to [threshold, max_value] with slope given by exponent.
	Args:
		x: Input tensor.
		exponent: In nonlinear regime (away from x=0), the output varies by this
		factor for every change of x by 1.0.
		max_value: Limiting value at x=inf.
		threshold: Limiting value at x=-inf. Stablizes training when outputs are
		pushed to 0.
	Returns:
		A tensor with pointwise nonlinearity applied.
	"""
	x = tf_float32(x)
	return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold

def crop_and_compensate_delay(audio: tf.Tensor, audio_size: int, ir_size: int,
							  padding: Text,
							  delay_compensation: int) -> tf.Tensor:
	"""Crop audio output from convolution to compensate for group delay.
	Args:
		audio: Audio after convolution. Tensor of shape [batch, time_steps].
		audio_size: Initial size of the audio before convolution.
		ir_size: Size of the convolving impulse response.
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		ir_timesteps - 1).
		delay_compensation: Samples to crop from start of output audio to compensate
		for group delay of the impulse response. If delay_compensation < 0 it
		defaults to automatically calculating a constant group delay of the
		windowed linear phase filter from frequency_impulse_response().
	Returns:
		Tensor of cropped and shifted audio.
	Raises:
		ValueError: If padding is not either 'valid' or 'same'.
	"""
	# Crop the output.
	if padding == 'valid':
		crop_size = ir_size + audio_size - 1
	elif padding == 'same':
		crop_size = audio_size
	else:
		raise ValueError('Padding must be \'valid\' or \'same\', instead '
						'of {}.'.format(padding))

	# Compensate for the group delay of the filter by trimming the front.
	# For an impulse response produced by frequency_impulse_response(),
	# the group delay is constant because the filter is linear phase.
	total_size = int(audio.shape[-1])
	crop = total_size - crop_size
	start = ((ir_size - 1) // 2 -
			1 if delay_compensation < 0 else delay_compensation)
	end = crop - start
	return audio[:, start:-end]




def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
	"""Calculate final size for efficient FFT.
	Args:
		frame_size: Size of the audio frame.
		ir_size: Size of the convolving impulse response.
		power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
		numbers. TPU requires power of 2, while GPU is more flexible.
	Returns:
		fft_size: Size for efficient FFT.
	"""
	convolved_frame_size = ir_size + frame_size - 1
	if power_of_2:
		# Next power of 2.
		fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
	else:
		fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
	return fft_size

def fft_convolve(audio: tf.Tensor,
				 impulse_response: tf.Tensor,
				 padding: Text = 'same',
				 delay_compensation: int = -1) -> tf.Tensor:
	"""Filter audio with frames of time-varying impulse responses.
	Time-varying filter. Given audio [batch, n_samples], and a series of impulse
	responses [batch, n_frames, n_impulse_response], splits the audio into frames,
	applies filters, and then overlap-and-adds audio back together.
	Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
	convolution for large impulse response sizes.
	Args:
		audio: Input audio. Tensor of shape [batch, audio_timesteps].
		impulse_response: Finite impulse response to convolve. Can either be a 2-D
		Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
		ir_frames, ir_size]. A 2-D tensor will apply a single linear
		time-invariant filter to the audio. A 3-D Tensor will apply a linear
		time-varying filter. Automatically chops the audio into equally shaped
		blocks to match ir_frames.
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		ir_timesteps - 1).
		delay_compensation: Samples to crop from start of output audio to compensate
		for group delay of the impulse response. If delay_compensation is less
		than 0 it defaults to automatically calculating a constant group delay of
		the windowed linear phase filter from frequency_impulse_response().
	Returns:
		audio_out: Convolved audio. Tensor of shape
			[batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
			[batch, audio_timesteps] ('same' padding).
	Raises:
		ValueError: If audio and impulse response have different batch size.
		ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
		number of impulse response frames is on the order of the audio size and
		not a multiple of the audio size.)
	"""
	audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

	# Get shapes of audio.
	batch_size, audio_size = audio.shape.as_list()

	# Add a frame dimension to impulse response if it doesn't have one.
	ir_shape = impulse_response.shape.as_list()
	if len(ir_shape) == 2:
		impulse_response = impulse_response[:, tf.newaxis, :]

	# Broadcast impulse response.
	if ir_shape[0] == 1 and batch_size > 1:
		impulse_response = tf.tile(impulse_response, [batch_size, 1, 1])

	# Get shapes of impulse response.
	ir_shape = impulse_response.shape.as_list()
	batch_size_ir, n_ir_frames, ir_size = ir_shape

	# Validate that batch sizes match.
	if batch_size != batch_size_ir:
		raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
						'be the same.'.format(batch_size, batch_size_ir))

	# Cut audio into frames.
	frame_size = int(np.ceil(audio_size / n_ir_frames))
	hop_size = frame_size
	audio_frames = tf.signal.frame(audio, frame_size, hop_size, pad_end=True)

	# Check that number of frames match.
	n_audio_frames = int(audio_frames.shape[1])
	if n_audio_frames != n_ir_frames:
		raise ValueError(
			'Number of Audio frames ({}) and impulse response frames ({}) do not '
			'match. For small hop size = ceil(audio_size / n_ir_frames), '
			'number of impulse response frames must be a multiple of the audio '
			'size.'.format(n_audio_frames, n_ir_frames))

	# Pad and FFT the audio and impulse responses.
	fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
	audio_fft = tf.signal.rfft(audio_frames, [fft_size])
	ir_fft = tf.signal.rfft(impulse_response, [fft_size])

	# Multiply the FFTs (same as convolution in time).
	audio_ir_fft = tf.multiply(audio_fft, ir_fft)
	# print(f"inspecting inside: audio fft shape-->{audio_fft.shape},ir fft shape-->{ir_fft.shape}, audio_ir_fft-->{audio_ir_fft.shape}")
	# Take the IFFT to resynthesize audio.
	audio_frames_out = tf.signal.irfft(audio_ir_fft)
	audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)

	# Crop and shift the output audio.
	return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
									delay_compensation)

def apply_window_to_impulse_response(impulse_response: tf.Tensor,
									 window_size: int = 0,
									 causal: bool = False) -> tf.Tensor:
	"""Apply a window to an impulse response and put in causal form.
	Args:
		impulse_response: A series of impulse responses frames to window, of shape
		[batch, n_frames, ir_size].
		window_size: Size of the window to apply in the time domain. If window_size
		is less than 1, it defaults to the impulse_response size.
		causal: Impulse responnse input is in causal form (peak in the middle).
	Returns:
		impulse_response: Windowed impulse response in causal form, with last
		dimension cropped to window_size if window_size is greater than 0 and less
		than ir_size.
	"""
	impulse_response = tf_float32(impulse_response)

	# If IR is in causal form, put it in zero-phase form.
	if causal:
		impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

	# Get a window for better time/frequency resolution than rectangular.
	# Window defaults to IR size, cannot be bigger.
	ir_size = int(impulse_response.shape[-1])
	if (window_size <= 0) or (window_size > ir_size):
		window_size = ir_size
	window = tf.signal.hann_window(window_size)

	# Zero pad the window and put in in zero-phase form.
	padding = ir_size - window_size
	if padding > 0:
		half_idx = (window_size + 1) // 2
		window = tf.concat([window[half_idx:],
							tf.zeros([padding]),
							window[:half_idx]], axis=0)
	else:
		window = tf.signal.fftshift(window, axes=-1)

	# Apply the window, to get new IR (both in zero-phase form).
	window = tf.broadcast_to(window, impulse_response.shape)
	impulse_response = window * tf.math.real(impulse_response)

	# Put IR in causal form and trim zero padding.
	if padding > 0:
		first_half_start = (ir_size - (half_idx - 1)) + 1
		second_half_end = half_idx + 1
		impulse_response = tf.concat([impulse_response[..., first_half_start:],
									impulse_response[..., :second_half_end]],
									axis=-1)
	else:
		impulse_response = tf.signal.fftshift(impulse_response, axes=-1)

	return impulse_response
def frequency_impulse_response(magnitudes: tf.Tensor,
							   window_size: int = 0) -> tf.Tensor:
	"""Get windowed impulse responses using the frequency sampling method.
	Follows the approach in:
	https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html
	Args:
		magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
		n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
		last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
		f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
		audio into equally sized frames to match frames in magnitudes.
		window_size: Size of the window to apply in the time domain. If window_size
		is less than 1, it defaults to the impulse_response size.
	Returns:
		impulse_response: Time-domain FIR filter of shape
		[batch, frames, window_size] or [batch, window_size].
	Raises:
		ValueError: If window size is larger than fft size.
	"""
	# Get the IR (zero-phase form).
	magnitudes = tf.complex(magnitudes, tf.zeros_like(magnitudes)) #imaginary part == 0
	impulse_response = tf.signal.irfft(magnitudes)

	# Window and put in causal form.
	impulse_response = apply_window_to_impulse_response(impulse_response,
														window_size)

	return impulse_response



# use this one when istft is fixed!
def _istft_tensorflow(stfts, hparams):
    return tf.signal.inverse_stft(
        stfts, hparams.win_length, hparams.hop_length, hparams.n_fft
    )

def _stft_tensorflow(signals, hparams):
    return tf.signal.stft(
        signals,
        hparams.win_length,
        hparams.hop_length,
        hparams.n_fft,
        pad_end=False,
    )



def _griffin_lim_tensorflow(S, hparams):
	"""TensorFlow implementation of Griffin-Lim
	Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb and
	https://github.com/keithito/tacotron/blob/master/util/audio.py
	issue: https://github.com/tensorflow/tensorflow/issues/28444
	"""
	# S = tf.expand_dims(S, 0)
	# print("spec shape", S.shape)
	S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
	y = _istft_tensorflow(S_complex, hparams)
	for _ in range(hparams.griffin_lim_iters):
		est = _stft_tensorflow(y, hparams)
		# print("recon shape", est.shape)
		angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
		y = _istft_tensorflow(S_complex * angles, hparams)
	# return tf.squeeze(y, 0)
	return y
def _istft(y, hparams):
    return librosa.istft(
        y, hop_length=hparams.hop_length, win_length=hparams.win_length
    )
def _stft(y, hparams):
    return librosa.stft(
        y=y,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length    )

def _griffin_lim_numpy(S, hparams):
	S = S[0]
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles, hparams)
	# print(f"S:{S.shape},S_com:{S_complex.shape}, y:{y.shape}, angle:{angles.shape}")

	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y, hparams)))
		# print("angles", _stft(y, hparams).shape)
		y = _istft(S_complex * angles, hparams)
	y = tf.expand_dims(y, 0)
	return y


class compressor_smoothing_gain_cell(tf.keras.layers.Layer):
	def __init__(self,sr, attack_init, release_init, scaling_factor = 1.0):
		super(compressor_smoothing_gain_cell, self).__init__()
		self.attack_time_init=tf.math.exp(-log10(9.0)/(sr*attack_init*1.0E-3*scaling_factor))
		self.release_time_init=tf.math.exp(-log10(9.0)/(sr*release_init*1.0E-3*scaling_factor))
	def build(self, input_shape):
		self.attack_time = tf.Variable(name="attack_time_constant", initial_value=self.attack_time_init, trainable=True)
		self.release_time = tf.Variable(name="release_time_constant", initial_value=self.release_time_init, trainable=True)

	def call(self,inp,prev_sample):
		# if self.attack_time is None:
		# 	self.attack_time = tf.Variable(name="attack_time_constant", initial_value=self.attack_time_init, trainable=True)
		# if self.release_time is None:
		# 	self.release_time = tf.Variable(name="release_time_constant", initial_value=self.release_time_init, trainable=True)
		out = tf.where(inp>=prev_sample, self.attack_time * prev_sample + (1-self.attack_time) * inp,  self.release_time * prev_sample + (1-self.release_time) * inp)
		# print(out)
		# print("inp", inp.numpy(), prev_sample.numpy(),out)
		# if prev_sample>=inp:
		# 	out = self.attack_time * prev_sample + (1-self.attack_time) * inp
		# else:
		# 	out = self.release_time * prev_sample + (1-self.release_time) * inp   
		return out

def compressor_smoothing_gain_cell_func(inp,prev_sample, attack_time, release_time):
	out = tf.where(inp>=prev_sample, attack_time * prev_sample + (1-attack_time) * inp, release_time * prev_sample + (1-release_time) * inp)
	return out


class HParams(object):
	""" Hparams was removed from tf 2.0alpha so this is a placeholder
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
# @tf.function()
def compressor_fft_based(audio,sr,threshold, ratio,makeup, attack_time,  release_time, spec_gain_floor, if_ori_phase=True):
	

	
	audio = tf_float32(audio)
	#convert audio to spectrogram
	spectrogram_raw = _stft_tensorflow(audio, hparams)
	# spectrogram_raw = tf.signal.stft(audio, frame_length = 512, frame_step = 128, fft_length = 1024) #frame number = (total_len-frame_len)/hop_len + 1, freq_bin = fft_len/2+1
	# spectrogram = tf_float32(spectrogram_raw)
	spectrogram_mag = tf_float32(tf.abs(spectrogram_raw))
	spectrogram_phase = tf_float32(tf.math.angle(spectrogram_raw))

	spectrogram_db = 20*log10(abs(spectrogram_mag)) 
	# print("spectrogram_db", spectrogram_db.shape, spectrogram_db)
	# print("raw vs float", spectrogram_raw, spectrogram_mag)
	# print("phase", spectrogram_phase)
	# plt.imsave("spec_ori.png",np.transpose(spectrogram_db[0].numpy()))
	plt.imsave("spec_ori.png",np.rot90(spectrogram_db[0].numpy(), k = 1))

	#compute gain based on threshold and ratio
	compressed_spectrogram_db = tf.where(tf.math.greater(spectrogram_db, threshold), 
								threshold+(spectrogram_db-threshold)/ratio,spectrogram_db)
	gain=tf.math.subtract(compressed_spectrogram_db,spectrogram_db)
	cell = compressor_smoothing_gain_cell(sr, attack_time, release_time, scaling_factor = compressed_spectrogram_db.shape[1])
	#smooth gain
	prev_sample = gain[:, 0, :]
	smoothed_gain = tf.TensorArray(dtype=tf.float32, size=spectrogram_db.shape[1])
	for i in range(gain.shape[1]): 
			prev_sample = cell(gain[:,i,:], prev_sample)
			smoothed_gain = smoothed_gain.write(i, prev_sample)
	smoothed_gain = smoothed_gain.stack()
	smoothed_gain = tf.where(tf.math.greater(spec_gain_floor, smoothed_gain), 
								spec_gain_floor,smoothed_gain)
	smoothed_gain = tf.transpose(smoothed_gain, perm=[1, 0, 2])
	# print("check gain", gain.numpy(), smoothed_gain.numpy())
	# for i in range(20):
	# 	print("smoothed gain",gain[0, i, :10].numpy(), smoothed_gain[0, i, :10].numpy())
	plt.imsave("smoothed_gain.png",np.rot90(smoothed_gain[0].numpy()) , dpi=1200)
	plt.imsave("gain.png",np.rot90(gain[0].numpy()) , dpi=1200)

	#add gain to sig
	smoothed_compressed_spectrogram_db = spectrogram_db + smoothed_gain + makeup
	plt.imsave("compressed_spec.png",np.rot90(smoothed_compressed_spectrogram_db[0].numpy()) , dpi=1200)
	smoothed_compressed_spectrogram = 10.0**((smoothed_compressed_spectrogram_db)/20.0)
	# print("dbed smoothed conp",tf.equal(smoothed_compressed_spectrogram,spectrogram_mag).numpy().all(), ((smoothed_compressed_spectrogram-spectrogram_mag)>1e-2).numpy().any())

	#inverse stft to return sig #https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/7.0-Tensorflow-spectrograms-and-inversion.ipynb
	if if_ori_phase:
		# phase_final = spectrogram_phase
		out = mag_phase_2_real_imag(smoothed_compressed_spectrogram, spectrogram_phase) #pending change
		# out = mag_phase_2_real_imag(10.0**((compressed_spectrogram_db)/20.0), spectrogram_phase)
		compressed_sig = _istft_tensorflow(out, hparams)
		# compressed_sig = tf.signal.inverse_stft(out,frame_length = 512, frame_step = 128, fft_length = 1024, window_fn=tf.signal.inverse_stft_window_fn(32))
	else:

		# compressed_sig = _griffin_lim_tensorflow(smoothed_compressed_spectrogram,hparams)
		# print("audio", audio.shape, smoothed_compressed_spectrogram.shape)
		input_to_grif = tf.transpose(smoothed_compressed_spectrogram[0]).numpy()
		compressed_sig = librosa.griffinlim(input_to_grif, n_iter=hparams.griffin_lim_iters, hop_length=hparams.hop_length, win_length=hparams.win_length, window='hann', center=False)
		compressed_sig = tf.convert_to_tensor(compressed_sig[tf.newaxis, ...])
		# print("compressed sig shape", compressed_sig.shape)
		# compressed_sig = _griffin_lim_numpy(smoothed_compressed_spectrogram,hparams)
	#mag-phase 2 real-img


	# for x, y in zip(out[0], spectrogram_raw[0]):
	# 	for a, b in zip(x, y):
	# 		if tf.abs(a-b)>1e-3:
		
	# 			print(a, b)
	# print(f"compressed sig, {out.shape, spectrogram_raw.shape},{out==spectrogram_raw},{compressed_sig, audio}")
	return compressed_sig
	#     prev_sample = cell(gain_sample, prev_sample)
	#     smoothed_gain.append(prev_sample)
	# smoothed_gain = tf.convert_to_tensor(smoothed_gain)
	# print(f"without TBPT TIME:{time.time()-start:.2f}" )
	#smooth gain based on attack and release

	#add makeup gain


def get_down_sampled(inp, hparams):
	inp = inp[..., tf.newaxis]
	avg_kernel = tf.convert_to_tensor([1/hparams.win_length for _ in range (hparams.win_length)])[:,tf.newaxis, tf.newaxis] #filter width, in_channel, out_channel
	out_seq = tf.nn.conv1d(inp, avg_kernel, stride = hparams.hop_length, padding = "VALID")
	# for x in out_seq[0,:,0]:
	# 	print(x.numpy(), "sss!")
	# print("outseq",out_seq)
	return out_seq



def compressor_beta(audio,sr,threshold, ratio,makeup, attack_time,  release_time, spec_gain_floor, if_ori_phase=True):
	
	audio = tf_float32(audio)
	# audio_sign = tf.where(audio<0)
	audio_db = 20*log10(abs(audio)) 


	# spectrogram_phase = tf_float32(tf.math.angle(spectrogram_raw))
	# plt.imsave("spec.png",np.transpose(spectrogram_db[0].numpy()))
	
	#compute gain based on threshold and ratio
	compressed_audio_db = tf.where(tf.math.greater(audio_db, threshold), 
								threshold+(audio_db-threshold)/ratio,audio_db)
	gain=tf.math.subtract(compressed_audio_db,audio_db) #[batch, length]
	x = [i for i in range(gain.shape[1])] 
	y = gain[0]
	plt.plot(x, y, color ="red")
	# print("gain",gain)
	plt.savefig('gain_time.png')
	# plt.imsave("gain_time.png",gain.numpy() , dpi=1200)

	#try to get gain from spec
	spectrogram_raw = _stft_tensorflow(audio, hparams)
	spectrogram_mag = tf_float32(tf.abs(spectrogram_raw))
	spectrogram_phase = tf_float32(tf.math.angle(spectrogram_raw))

	spectrogram_db = 20*log10(abs(spectrogram_mag)) 
	compressed_spectrogram_db = tf.where(tf.math.greater(spectrogram_db, threshold), 
								threshold+(spectrogram_db-threshold)/ratio,spectrogram_db)
	gain_fft=tf.math.subtract(compressed_spectrogram_db,spectrogram_db)
	cell = compressor_smoothing_gain_cell(sr, attack_time, release_time, scaling_factor = compressed_spectrogram_db.shape[1])
	#smooth gain
	prev_sample = gain_fft[:, 0, :]
	smoothed_gain_fft = tf.TensorArray(dtype=tf.float32, size=spectrogram_db.shape[1])
	for i in range(gain_fft.shape[1]): 
			prev_sample = cell(gain_fft[:,i,:], prev_sample)
			smoothed_gain_fft = smoothed_gain_fft.write(i, prev_sample)
	smoothed_gain_fft = smoothed_gain_fft.stack()
	smoothed_gain_fft = tf.where(tf.math.greater(spec_gain_floor, smoothed_gain_fft), 
								spec_gain_floor,smoothed_gain_fft)
	smoothed_gain_fft = tf.transpose(smoothed_gain_fft, perm=[1, 0, 2]) #batch, t, f

	plt.imsave("smoothed_gain.png",np.transpose(smoothed_gain_fft[0].numpy()) , dpi=1200)


	out = mag_phase_2_real_imag(smoothed_gain_fft, spectrogram_phase) #pending change
	# print("smoothed gain fft", smoothed_gain_fft.shape, spectrogram_phase.shape, out.shape)

	smoothed_gain_fft_time = _istft_tensorflow(out, hparams)
	x = [i for i in range(smoothed_gain_fft_time.shape[1])] 

	y = smoothed_gain_fft_time[0]
	plt.plot(x, y, color ="red")
	plt.savefig('gain_freq.png')




	smoothed_compressed_audio_db = audio_db + gain + makeup
	smoothed_compressed_audio = 10.0**((smoothed_compressed_audio_db)/20.0)
	smoothed_compressed_audio = tf.where(audio<0, -smoothed_compressed_audio, smoothed_compressed_audio)
	# print("check audio", smoothed_compressed_audio_db.shape, audio.shape, smoothed_compressed_audio_db, smoothed_compressed_audio)
	return smoothed_compressed_audio


	"""cell = compressor_smoothing_gain_cell(sr, attack_time, release_time)
	#smooth gain
	prev_sample = gain[:, 0]
	smoothed_gain = tf.TensorArray(dtype=tf.float32, size=compressed_audio_db.shape[1])
	for i in range(gain.shape[1]): 
			prev_sample = cell(gain[:,i], prev_sample)
			smoothed_gain = smoothed_gain.write(i, prev_sample)
	smoothed_gain = smoothed_gain.stack()
	print("smoothed gain",smoothed_gain.shape)
	# smoothed_gain = tf.where(tf.math.greater(spec_gain_floor, smoothed_gain), 
	# 							spec_gain_floor,smoothed_gain)
	# smoothed_gain = tf.transpose(smoothed_gain, perm=[1, 0, 2])
	# print("check gain", gain.numpy(), smoothed_gain.numpy().shape)
	# for i in range(20):
	# 	print("smoothed gain",gain[0, i, :10].numpy(), smoothed_gain[0, i, :10].numpy())
	plt.imsave("smoothed_gain.png",np.transpose(smoothed_gain[0].numpy()) , dpi=1200)
	plt.imsave("gain.png",np.transpose(gain[0].numpy()) , dpi=1200)

	#add gain to sig
	smoothed_compressed_spectrogram_db = spectrogram_db + smoothed_gain + makeup
	plt.imsave("smoothed_spec.png",np.transpose(smoothed_compressed_spectrogram_db[0].numpy()) , dpi=1200)
	smoothed_compressed_spectrogram = 10.0**((smoothed_compressed_spectrogram_db)/20.0)
	# print("dbed smoothed conp",tf.equal(smoothed_compressed_spectrogram,spectrogram_mag).numpy().all(), ((smoothed_compressed_spectrogram-spectrogram_mag)>1e-2).numpy().any())

	#inverse stft to return sig #https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/7.0-Tensorflow-spectrograms-and-inversion.ipynb
	if if_ori_phase:
		# phase_final = spectrogram_phase
		out = mag_phase_2_real_imag(smoothed_compressed_spectrogram, spectrogram_phase) #pending change
		# out = mag_phase_2_real_imag(10.0**((compressed_spectrogram_db)/20.0), spectrogram_phase)
		compressed_sig = _istft_tensorflow(out, hparams)
		# compressed_sig = tf.signal.inverse_stft(out,frame_length = 512, frame_step = 128, fft_length = 1024, window_fn=tf.signal.inverse_stft_window_fn(32))
	else:

		# compressed_sig = _griffin_lim_tensorflow(smoothed_compressed_spectrogram,hparams)
		print("audio", audio.shape, smoothed_compressed_spectrogram.shape)
		input_to_grif = tf.transpose(smoothed_compressed_spectrogram[0]).numpy()
		compressed_sig = librosa.griffinlim(input_to_grif, n_iter=hparams.griffin_lim_iters, hop_length=hparams.hop_length, win_length=hparams.win_length, window='hann', center=False)
		compressed_sig = tf.convert_to_tensor(compressed_sig[tf.newaxis, ...])
		print("compressed sig shape", compressed_sig.shape)
		# compressed_sig = _griffin_lim_numpy(smoothed_compressed_spectrogram,hparams)
	#mag-phase 2 real-img


	# for x, y in zip(out[0], spectrogram_raw[0]):
	# 	for a, b in zip(x, y):
	# 		if tf.abs(a-b)>1e-3:
		
	# 			print(a, b)
	# print(f"compressed sig, {out.shape, spectrogram_raw.shape},{out==spectrogram_raw},{compressed_sig, audio}")
	return compressed_sig
	#     prev_sample = cell(gain_sample, prev_sample)
	#     smoothed_gain.append(prev_sample)
	# smoothed_gain = tf.convert_to_tensor(smoothed_gain)
	# print(f"without TBPT TIME:{time.time()-start:.2f}" )
	#smooth gain based on attack and release

	#add makeup gain
	"""

def frequency_filter(audio: tf.Tensor,
						magnitudes: tf.Tensor,
						window_size: int = 0,
						padding: Text = 'same') -> tf.Tensor:
	"""Filter audio with a finite impulse response filter.
	Args:
		audio: Input audio. Tensor of shape [batch, audio_timesteps].
		magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
		n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
		last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
		f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
		audio into equally sized frames to match frames in magnitudes.
		window_size: Size of the window to apply in the time domain. If window_size
		is less than 1, it is set as the default (n_frequencies).
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		window_size - 1).
	Returns:
		Filtered audio. Tensor of shape
			[batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
			[batch, audio_timesteps] ('same' padding).
	"""
	impulse_response = frequency_impulse_response(magnitudes,
													window_size=window_size)

	return fft_convolve(audio, impulse_response, padding=padding)
def compressor_time_averaged(audio,sr,threshold, ratio,makeup, attack_time,  release_time, downsample_factor = 16.0, if_save_fig = False):
	
	audio = tf_float32(audio)
	# audio_sign = tf.where(audio<0)
	audio_db = 20*log10(abs(audio)+1e-8) 
	# print("aud",audio,sr,threshold, ratio,makeup, attack_time,  release_time,)
	#compute gain based on threshold and ratio
	compressed_audio_db = tf.where(tf.math.greater(audio_db, threshold), 
								threshold+(audio_db-threshold)/ratio,audio_db)
	gain=tf.math.subtract(compressed_audio_db,audio_db) #[batch, length]
	
	# gain_downsampled = get_down_sampled(gain, hparams)
	# gain_downsampled = downsample_with_windows(gain, hparams)
	# gain_downsampled = resample(gain, 1377, method = 'linear')[..., tf.newaxis] #['nearest', 'linear', 'cubic', 'window']
	gain_downsampled = resample(gain, int(gain.shape[1]/downsample_factor), method = 'linear')[..., tf.newaxis] #['nearest', 'linear', 'cubic', 'window']

	# gain_downsampled = tf.convert_to_tensor(scipy.signal.resample(gain.numpy(), 1377, axis = -1))[..., tf.newaxis]

	# print("check!!!",gain_downsampled.shape, gain.shape)
	#smooth gain

	# cell = compressor_smoothing_gain_cell(sr, attack_time, release_time, scaling_factor = int(downsample_factor))

	#smooth gain
	prev_sample = gain_downsampled[:, 0, :]
	smoothed_gain = tf.TensorArray(dtype=tf.float32, size=gain_downsampled.shape[1])



	# for i in range(gain_downsampled.shape[1]): 
	# 	prev_sample = cell(gain_downsampled[:,i,:], prev_sample)
	# 	smoothed_gain = smoothed_gain.write(i, prev_sample)
	# gain_downsampled_smoothed = tf.transpose(smoothed_gain.stack(), perm = [1, 0, 2])




	def condition(i, out_seq, prev_sample):
		return tf.less(i, gain_downsampled.shape[1])
	def body(i, out_seq, prev_sample):
		prev_sample = compressor_smoothing_gain_cell_func(gain_downsampled[:,i,:],prev_sample, attack_time, release_time)
		
		# prev_sample= cell(gain_downsampled[:,i,:],prev_sample)
		out_seq = out_seq.write(i,prev_sample)
		return i+1, out_seq, prev_sample
	# out_seq = tf.TensorArray(dtype=tf.float32, size=file_len)
	# smoothed_gain.write(0, prev_sample)
	_, smoothed_gain, _ = tf.while_loop(condition, body, [0, smoothed_gain, prev_sample])
	gain_downsampled_smoothed = tf.transpose(smoothed_gain.stack(), perm = [1, 0, 2])



	#pending change how to align
	# gain_downsampled_smoothed_upsampled = upsample_with_windows(gain_downsampled_smoothed, (1+gain.shape[1]//gain_downsampled_smoothed.shape[1])*gain_downsampled_smoothed.shape[1])
	gain_downsampled_smoothed_upsampled = upsample_with_windows(gain_downsampled_smoothed, int(gain_downsampled_smoothed.shape[1]*downsample_factor))

	# print(f"gain,{gain.shape},{gain_downsampled.shape},{gain_downsampled_smoothed_upsampled.shape}")

	if if_save_fig:
		ax1 = plt.subplot(2,2, 1)
		x = [i for i in range(gain.shape[1])] 
		y = gain[0]
		ax1.set_title(f"oringal gain: {len(y)}")
		plt.axis('off')
		plt.plot(x, y, color ="red")
		
		ax2 = plt.subplot(2,2, 2)
		factor = int(gain.shape[1]/gain_downsampled.shape[1])
		x = [i*factor for i in range(gain_downsampled.shape[1])]
		y = gain_downsampled[0, :, 0]
		ax2.set_title(f"gain_ds: {len(y)}")
		plt.axis('off')
		plt.plot(x, y, color ="blue")


		ax3 = plt.subplot(2,2, 3)
		x = [i*factor for i in range(gain_downsampled_smoothed.shape[1])] 
		y = gain_downsampled_smoothed[0]
		ax3.set_title(f"gain_ds_sm: {len(y)}")
		plt.axis('off')
		plt.plot(x, y, color ="yellow")

		ax4 = plt.subplot(2,2, 4)
		x = [i for i in range(gain_downsampled_smoothed_upsampled.shape[1])] 
		y = gain_downsampled_smoothed_upsampled[0]
		ax4.set_title(f"gain_ds_sm_us: {len(y)}")
		plt.axis('off')
		plt.plot(x, y, color ="black")

		plt.savefig('gain_reduced.png', dpi = 1200)

	smoothed_compressed_audio_db = audio_db + gain_downsampled_smoothed_upsampled[:, :gain.shape[1], 0] + makeup
	smoothed_compressed_audio = 10.0**((smoothed_compressed_audio_db)/20.0)
	smoothed_compressed_audio = tf.where(audio<0, -smoothed_compressed_audio, smoothed_compressed_audio)
	# print("check audio", smoothed_compressed_audio_db.shape, audio.shape, smoothed_compressed_audio_db, smoothed_compressed_audio)
	return smoothed_compressed_audio
def distortion_actan(audio, threshold_dist, if_save_fig= False):
	#should I normalize threshold_dist
	#https://www.vicanek.de/articles/WaveshaperHarmonix.pdf
	audio = tf_float32(audio)
	if if_save_fig:
		x = tf.convert_to_tensor([i/100 for i in range(-100, 100)])[tf.newaxis, ...]
		y = (2/math.pi) * tf.math.atan( (0.5*math.pi*threshold_dist) * x)
		plt.plot(x.numpy()[0], y.numpy()[0], color ="red")
		plt.savefig(f'dist_gain{threshold_dist}.png')

	return (2/math.pi) * tf.math.atan( (0.5*math.pi*threshold_dist) * audio)



if __name__=="__main__":
	
	threshold_init = -20.0
	ratio_init = 8.0
	makeup_init = 0.0
	attack_init = 1e-3
	release_init = 1e-3
	spec_gain_floor = -30
	# attack_init = 1
	# release_init = 1
	# sample_rate = 16000
	audio_path = "../archive/Drums1_unprocessed.wav"
	audio,sample_rate = sf.read(audio_path)
	audio = audio[np.newaxis, ...]

	hparams = HParams(  
		# spectrogramming
		win_length = 256,
		n_fft = 1024,
		hop_length= 128,
		ref_level_db = 50,
		min_level_db = -100,
		# mel scaling
		num_mel_bins = 128,
		mel_lower_edge_hertz = 0,
		mel_upper_edge_hertz = 10000,
		# inversion
		power = 1.5, # for spectral inversion
		griffin_lim_iters = 1200,
		pad=True,
		#
	)

	compressed_sig = compressor_time_averaged(audio,sample_rate,threshold_init, ratio_init, makeup_init,attack_init,  release_init, if_save_fig= True)	
	sf.write("Drums1_compressed_time_avg.wav", compressed_sig[0].numpy(), sample_rate) 
