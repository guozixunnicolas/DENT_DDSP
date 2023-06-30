import glob 
import os 
import soundfile as sf
import numpy as np
# training_datas = glob.glob("/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/org/data_src_train_8k/data/**/*.wav")
training_datas = glob.glob("/home/emrys/G/zixun/espnet/egs2/asr_rl/clean/*.wav")

method = "g726"
# method = "add_rats_noise"
# method = "codec2"
rat_noise, _ = sf.read("/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/rats_noise.wav")
sr = 8000
def augment(audio_path, method, outname=None):
	audio_array, sr = sf.read(audio_path) 
	if method == "add_rats_noise":
		return audio_array+rat_noise[:len(audio_array)]
	if method == "g726":
		inter1 = "inter1.wav"
		inter2 = "inter2.wav"
		inter3 = "inter3.wav"
		command1 = f"ffmpeg -hide_banner -loglevel error -i {audio_path} -ar 8k -y {inter1}"

		command2 = f"ffmpeg -hide_banner -loglevel error -i {inter1} -acodec g726 -b:a 16k {inter2}"

		command3 = f"ffmpeg -hide_banner -loglevel error -i {inter2} -ar 8k -y {inter3}"

		command4 = f"rm {inter1} {inter2} {inter3}"

		os.system(command1)
		os.system(command2)
		os.system(command3)
		audio_array_g726, sr = sf.read(inter3) 
		os.system(command4)
		if len(audio_array_g726)>len(audio_array):
			audio_array_g726 = audio_array_g726[:len(audio_array)]
			print("padded")
		elif len(audio_array_g726)<len(audio_array):
			audio_array_g726 = audio_array
			print("werid")

		# print(audio_array_g726.shape,audio_array.shape )
		out = audio_array_g726+rat_noise[:len(audio_array)]
		return out 
	if method == "codec2":
		inter1 = "inter1.raw"
		inter2 = "test_4_5.bit"
		inter3 = "test_4_5.raw"
		inter4 = "test_4_5.wav"
		br = "700C"
		# br = "2400"

		command1 = f"ffmpeg -hide_banner -loglevel error -i {audio_path} -f s16le -ar 8k -acodec pcm_s16le {inter1}" # -ar 16k

		# /home/emrys/G/zixun/espnet/egs2/asr_rl/codec2/build_linux/src
		command2 = f"codec2/build_linux/src/c2enc {br} {inter1} {inter2}"
		# command2_ = f"codec2/build_linux/src/c2enc {br} test.raw {inter2}"

		command3 = f"codec2/build_linux/src/c2dec {br} {inter2} {inter3}"
		command4 = f"ffmpeg -hide_banner -loglevel error -f s16le -ar 8k -ac 1 -i {inter3} {inter4} "

		os.system(command1)
		os.system(command2)

		# if not os.path.exists("test.raw"):
		# 	os.system(command2)
		# else:
		# 	print("hha")
		# 	os.system(command2_)
		os.system(command3)
		os.system(command4)

		audio_array_g726, sr = sf.read(inter4) 
		command5 = f"rm {inter1} {inter2} {inter3} {inter4}"

		os.system(command5)

		if len(audio_array_g726)>len(audio_array):
			audio_array_g726 = audio_array_g726[:len(audio_array)]
			print("padded")
		elif len(audio_array_g726)<len(audio_array):
			audio_array_g726 = np.pad(audio_array_g726, (0, len(audio_array)-len(audio_array_g726)))
			print("werid")
		out = audio_array_g726+rat_noise[:len(audio_array)]
		# out = audio_array_g726

		return out 
	# 	cmd = f""



# for training_data in training_datas:
# 	new_dir = f"/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_src_train_8k_{method}/data/"+training_data.split("/")[-2] 
# 	if not os.path.exists(new_dir):
# 		os.makedirs(new_dir)
# 	new_file_name = new_dir+"/"+training_data.split("/")[-1] 
# 	print("processing input data", training_data)
# 	train_audio_aug = augment(training_data, method = method, outname = new_file_name)

# 	print("new dir", new_dir, new_file_name)
# 	print("saving", new_file_name)

# 	sf.write(new_file_name, train_audio_aug, sr,subtype='PCM_16')

for training_data in training_datas:
	new_dir = f"/home/emrys/G/zixun/espnet/egs2/asr_rl/clean_{method}"
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	new_file_name = new_dir+"/"+training_data.split("/")[-1] 
	print("processing input data", training_data)
	train_audio_aug = augment(training_data, method = method, outname = new_file_name)

	print("new dir", new_dir, new_file_name)
	print("saving", new_file_name)

	sf.write(new_file_name, train_audio_aug, sr,subtype='PCM_16')



# out = augment("/home/emrys/G/zixun/espnet/egs2/asr_rl/10280_eng_src_0-13.8140-18.620.wav", method)
# sf.write(f"paper_{method}.wav", out, sr,subtype='PCM_16')

# sf.read("out.wav")

