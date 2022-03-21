from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow_io as tfio
import soundfile as sf

class sine_creator(object):
    def __init__(self, dur, sr, amp=None):

        """
        gen = sine_creator(dur = 2, sr = 16000)
        gen([440, 880, 220])
        """
        self.dur = dur
        self.sr = sr
        self.amp = amp#between [0,1]
    def __call__(self, freqs = []):
        t = np.linspace(0., 1., self.dur*self.sr)
        if self.amp is None:
            self.amp = 1/len(freqs)
        sins = sum([self.amp*np.sin(2. * np.pi * f * t) for f in freqs])
        name = "_".join([str(x) for x in freqs])+".wav"
        write(name, self.sr, sins.astype(np.float32))

def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
	"""Differentiable stft in tensorflow, computed in batch."""
	assert frame_size * overlap % 2.0 == 0.0

	# Remove channel dim if present.
	audio = tf_float32(audio)
	if len(audio.shape) == 3:
		audio = tf.squeeze(audio, axis=-1)

	s = tf.signal.stft(
		signals=audio,
		frame_length=int(frame_size),
		frame_step=int(frame_size * (1.0 - overlap)),
		fft_length=int(frame_size),
		pad_end=pad_end)
	return s

def safe_log(x, eps=1e-5):
	"""Avoid taking the log of a non-positive number."""
	safe_x = tf.where(x <= eps, eps, x)
	return tf.math.log(safe_x)
def tf_float32(x):
	"""Ensure array/tensor is a float32 tf.Tensor."""
	if isinstance(x, tf.Tensor):
		return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
	else:
		return tf.convert_to_tensor(x, tf.float32)
def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
	mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
	return tf_float32(mag)

def compute_logmag(audio, size=2048, overlap=0.75, pad_end=True):
	return safe_log(compute_mag(audio, size, overlap, pad_end))

def specplot(audio,
			 vmin=-5,
			 vmax=1,
			 rotate=True,
			 size=512 + 256,
			 if_save = True,
			 fig_name ="spectro.png",
			 **matshow_kwargs):
	"""Plot the log magnitude spectrogram of audio."""
	# If batched, take first element.
	if len(audio.shape) == 2:
		audio = audio[0]

	logmag = compute_logmag(tf_float32(audio), size=size)
	if rotate:
		logmag = np.rot90(logmag)
  # Plotting.
	# plt.matshow(logmag,
	#             vmin=vmin,
	#             vmax=vmax,
	#             cmap=plt.cm.magma,
	#             aspect='auto',
				# **matshow_kwargs)
	plt.xticks([])
	plt.yticks([])
	plt.xlabel('Time')
	plt.ylabel('Frequency')
	if if_save:
		plt.imsave(fig_name,logmag , dpi=1200)
	plt.close()
def spec_gainplot(gain, fig_name = 'gain_time.png'):
	if gain.ndim==2:
		gain = np.repeat(gain[0][np.newaxis, ...], 200, axis = 0)
	elif gain.ndim==3:
		gain = gain[0]

	plt.matshow(np.rot90(gain), aspect='auto')	
	plt.savefig(fig_name)
def create_gauss_mag(n_seconds, sample_rate, frame_rate = 100, n_frequencies = 1000):

	n_frames = int(n_seconds * frame_rate)

	frequencies = np.linspace(0, sample_rate / 2.0, n_frequencies)

	lfo_rate = 0.5  # Hz
	n_cycles = n_seconds * lfo_rate
	center_frequency = 1000 + 500 * np.sin(np.linspace(0, 2.0*np.pi*n_cycles, n_frames))
	width = 500.0
	gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

	# Actually make the magnitudes.
	magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
	magnitudes = magnitudes[np.newaxis, ...]
	# for x in magnitudes[0]:
	# 	print(x)
	magnitudes /= magnitudes.max(axis=-1, keepdims=True)

	return magnitudes


def log10(x):
	numerator = tf.math.log(x)
	denominator = tf.math.log(10.0)
	return numerator / denominator
if __name__ == "__main__":
	audio_path = "../440.wav"
	# audio = tfio.audio.AudioIOTensor('gs://cloud-samples-tests/speech/brooklyn.flac')
	input_file, sr = sf.read(audio_path)
	input_file = input_file[np.newaxis, :]
	specplot(input_file)
	print("audii,", input_file.shape)

