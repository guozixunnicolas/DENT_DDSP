import soundfile as sf
import glob
import os
import numpy as np
def norm_one_file(audio_file):
    array, sr = sf.read(audio_file)
    # sf.write("tmp_unnorm.wav", array, 8000,subtype='PCM_16')
    normed_array = array/np.max(array) * 0.8
    return normed_array 
# f = "/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/data_Ach_test_8k/data/format.2/fe_03_1521-02110-B-000167-000348-A.wav"
# normed_array = norm_one_file(f)
# sf.write(normed_array, subtype="PCM_16")




#normalize /home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/data_Ach_test_8k --> test noisy
#normalize /home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/data_src_test_8k, train, valid -->clean 
#normalize /home/emrys/G/zixun/neural_noisy_speech/results_folder/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132/data_src_test_8k -->train, valid, noisy

# folders = ["/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/data_Ach_test_8k","/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/data_src_test_8k",\
#          "/home/emrys/G/zixun/neural_noisy_speech/results_folder/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132/data_src_train_8k", "/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/org/data_src_train_8k",\
#            "/home/emrys/G/zixun/neural_noisy_speech/results_folder/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132/data_src_valid_8k", "/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/org/data_src_valid_8k" ]


# folders = ["/home/emrys/G/zixun/Simu-GAN/Simu-GAN/results/data_src_test_8k", "/home/emrys/G/zixun/Simu-GAN/Simu-GAN/results/data_src_train_8k","/home/emrys/G/zixun/Simu-GAN/Simu-GAN/results/data_src_valid_8k"]
# folders = ["/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_Ach_valid_8k"]
# folders = ["/home/emrys/G/zixun/neural_noisy_speech/results_folder/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132--2.0/data_src_train_8k",\
# "/home/emrys/G/zixun/neural_noisy_speech/results_folder/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132-2.0/data_src_train_8k"]

# folders = ["/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_src_train_8k_add_rats_noise", "/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_src_train_8k_g726"]
folders = ["/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_Ach_train_8k", "/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_src_train_8k_codec2"]
for folder in folders:
    all_files = glob.glob(folder+"/**/**/*.wav")
    # print(len(all_files))
    new_parent_folder = "/".join(folder.split("/")[:-1])+ "/"+folder.split("/")[-1]+"_normed"
    print(new_parent_folder) #/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/data_src_test_8k_normed
    for f in all_files:
        sub_folder = "/".join(f.split("/")[-3:-1])
        if not os.path.exists(new_parent_folder+"/"+sub_folder):
            os.makedirs(new_parent_folder+"/"+sub_folder)
        name = f.split("/")[-1]
        new_dir = new_parent_folder+"/"+sub_folder+"/"+name

        normed_array = norm_one_file(f)
        sf.write(new_dir, normed_array, 8000,subtype='PCM_16')

        # print(f, "\n",new_dir)

# print(normed_array)