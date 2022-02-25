#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/12/28 16:40 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: Karan B, Sekhar Sahu S. An improved framework for Parkinson’s disease prediction using
#             Variational Mode Decomposition-Hilbert spectrum of speech signal.
#             Biocybernetics and Biomedical Engineering. 2021;41(2):717-32.
# @FileName : calcu_features.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V2.0 - ZL.Z：2022/2/4
#             V1.0 - ZL.Z：2021/12/28 - 2022/1/7
# 		      First version.
# @License  : None
# @Brief    : 基于分解的信号进行特征计算

from config import *
from util import read_csv
import datetime
import librosa
from librosa.core import spectrum
import parselmouth
from scipy.fft import dct
from scipy.stats import skew, kurtosis
from tftb.processing import inst_freq
from scipy.signal import hilbert
from pathos.pools import ProcessPool as Pool
from functools import partial


def calcu_mfcc(input_file):
	"""
	计算39维MFCC系数：13个MFCC特征（第一个系数为能量c0）及其对应的一阶和二阶差分
	:param input_file: 输入.wav音频文件，或是praat所支持的文件格式
	:return: 13*3维MFCC特征，每一列为一个MFCC特征向量 np.ndarray[shape=(39, n_frames), dtype=float64]
			 将上述13*3维MFCC特征计算为统计特征（均值/标准差/偏度/峰度）np.ndarray[shape=(39*4=156,), dtype=float64]
	"""
	sound = parselmouth.Sound(input_file)
	mfcc_obj = sound.to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=0.01,
	                         firstFilterFreqency=100.0, distance_between_filters=100.0)  # 默认额外包含c0
	mfcc_f = mfcc_obj.to_array()
	mfcc_delta1 = librosa.feature.delta(mfcc_f)  # 一阶差分
	mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)  # 二阶差分
	mfcc_f = np.vstack((mfcc_f, mfcc_delta1, mfcc_delta2))  # 整合成39维MFCC特征
	mfcc_mean = np.mean(mfcc_f, axis=1)
	mfcc_std = np.std(mfcc_f, axis=1)
	mfcc_skew = skew(mfcc_f, axis=1)
	mfcc_kurt = kurtosis(mfcc_f, axis=1)
	mfcc = np.hstack((mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt))
	return mfcc


def calcu_hcc(imfs: np.ndarray):
	"""
	计算基于信号分解的Hilbert cepstral coefficient (HCC) 特征,
	原文使用VMD，这里可选EMD/CEEMDAN/VMD
	Ref: Karan B, Sekhar Sahu S. An improved framework for Parkinson’s disease prediction using Variational
	Mode Decomposition-Hilbert spectrum of speech signal. Biocybernetics and Biomedical Engineering. 2021;41(2):717-32.
	:param imfs: 分解后的信号分量：对于EMD/CEEMDAN/VMD，为本征模态分量IMFs
	:return: HCC：1*信号分量数的两倍（瞬时能量+瞬时频率）, np.ndarray[shape=(信号分量数*2, ), dtype=float64]
	"""
	ieds, ifds = [], []
	for i_imf in imfs:  # 计算每个IMF的瞬时能量和瞬时频率对应的偏差IED和IFD
		analytic_sig = hilbert(i_imf)
		inst_amps = np.abs(analytic_sig)
		inst_freqs, timestamps = inst_freq(analytic_sig)
		diff_ie = [abs(j_ie - np.mean(inst_amps)) for j_ie in inst_amps]
		diff_if = [abs(j_if - np.mean(inst_freqs)) for j_if in inst_freqs]
		ied, ifd = sum(diff_ie) / len(inst_amps), sum(diff_if) / len(inst_freqs)
		ieds.append(np.where(ied == 0, np.finfo(np.float64).eps, ied))
		ifds.append(np.where(ifd == 0, np.finfo(np.float64).eps, ifd))
	ids = np.array(ieds + ifds)
	hcc = dct(np.log10(ids), type=2, norm="ortho")  # HCC特征
	return hcc


def calcu_decompose_mfcc(imfs: np.ndarray, input_file: str):
	"""
	计算基于信号分解的MFCC特征
	:param imfs: 分解后的信号分量：对于EMD/CEEMDAN/VMD，为本征模态分量IMFs
	:param input_file: 输入.wav音频文件，或是librosa所支持的文件格式
	:return: 39维decompose-MFCC特征，每一列为一个特征向量 np.ndarray[shape=(39, n_frames), dtype=float64]
			 将上述13*3维MFCC特征计算为统计特征（均值/标准差/偏度/峰度）np.ndarray[shape=(39*4=156,), dtype=float64]
	"""
	sig, sr = librosa.load(input_file, sr=None)
	stft_imfs = []
	for i_imf in imfs:  # 计算每个IMF的短时傅里叶变换
		i_imf_preem = librosa.effects.preemphasis(i_imf, coef=0.97)  # 预加重，系数0.97
		# NFFT=帧长=窗长=400个采样点(25ms,16kHz),窗移=帧移=2/5窗长=2/5*400=160个采样点(10ms,16kHz),汉明窗
		stft_imfs.append(spectrum.stft(i_imf_preem, n_fft=int(0.025*sr), hop_length=int(0.01*sr), window=np.hamming))
	stft_imfs.reverse()  # IMF1-N频率依次降低，因此需要反转
	stft_imfs_comp = np.array(stft_imfs).reshape(-1, np.array(stft_imfs).shape[-1])  # 频率从低到高合成频谱
	n_fft = 2 * (stft_imfs_comp.shape[0] - 1)  # 频谱合成后NFFT变化
	spectrogram_mag = np.abs(stft_imfs_comp)  # 幅值谱/振幅谱	print(spectrogram_mag.shape)
	spectrogram_pow = 1.0/n_fft * np.square(spectrogram_mag)  # 功率谱/功率谱密度PSD
	energy = np.sum(spectrogram_pow, 0)  # 储存每一帧的总能量
	energy = np.where(energy == 0, np.finfo(float).eps, energy)
	fb_mel = librosa.filters.mel(sr, n_fft, n_mels=26)  # Mel滤波器组的滤波器数量 = 26
	spectrogram_mel = np.dot(fb_mel, spectrogram_pow)  # 计算Mel谱
	spectrogram_mel = np.where(spectrogram_mel == 0, np.finfo(float).eps, spectrogram_mel)
	spec_mel_log = librosa.power_to_db(spectrogram_mel)  # 转换为log尺度
	# 前13个MFCC系数，升倒谱系数22, shape=(n_mfcc, t)=(13, 帧数n_frames), 每一列为一个特征向量
	mfcc_f = librosa.feature.mfcc(S=spec_mel_log, n_mfcc=13, lifter=22)  # log-mel谱DCT之后得到decompose-MFCC
	mfcc_f[0, :] = np.log10(energy)  # 将第0个系数替换成对数能量值
	mfcc_delta1 = librosa.feature.delta(mfcc_f)  # 一阶差分
	mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)  # 二阶差分
	decompose_mfcc = np.vstack((mfcc_f, mfcc_delta1, mfcc_delta2))  # 整合成39维特征
	dmfcc_mean = np.mean(decompose_mfcc, axis=1)
	dmfcc_std = np.std(decompose_mfcc, axis=1)
	dmfcc_skew = skew(decompose_mfcc, axis=1)
	dmfcc_kurt = kurtosis(decompose_mfcc, axis=1)
	dmfcc = np.hstack((dmfcc_mean, dmfcc_std, dmfcc_skew, dmfcc_kurt))
	return dmfcc


class GetFeatures:
	"""计算基于元音发音任务的各类特征"""
	def __init__(self, data_dir: str = '', decomp_dir: str = ''):
		"""
		初始化
		:param data_dir: 数据文件路径
		:param decomp_dir: 信号分解的数据主路径
		"""
		self.decomp_dir = decomp_dir
		self.subject_dir_list = []  # 每个被试的所有音频文件组成的列表
		self.decomposition_num = {'EMD': 6, 'CEEMDAN': 8, 'VMD': 4}
		for i_each_file in os.listdir(data_dir):
			if (i_each_file == 'WD') or (i_each_file == 'HC'):
				data_path_group = os.path.join(data_dir, i_each_file)
				for j_each_file in os.listdir(data_path_group):
					data_path_gender = os.path.join(data_path_group, j_each_file)
					for k_each_file in os.listdir(data_path_gender):
						self.subject_dir_list.append(os.path.join(data_path_gender, k_each_file))

	def get_features(self, subject_dir: str = '', feat: str = 'DMFCC'):
		"""
		获取对应信号的全部特征
        :param subject_dir: 被试主文件夹路径
		:param feat: 待提取特征'MFCC'/'DMFCC'/'HCC'
		:return: 所有特征集, pd.DataFrame类型
		"""
		try:
			print("---------- Processing %d / %d: ./%s ----------" %
				  (self.subject_dir_list.index(subject_dir) + 1, len(self.subject_dir_list),
				   os.path.relpath(subject_dir, os.getcwd())))
		except ValueError:
			pass
		group = os.path.normpath(subject_dir).split(os.sep)[-3]
		gender = os.path.normpath(subject_dir).split(os.sep)[-2]
		subject_id = os.path.normpath(subject_dir).split(os.sep)[-1].split("_")[1]
		csv_data = read_csv(subject_dir + "/" + subject_id + ".csv")
		name = csv_data[0][0].split("：")[1]
		age = int(csv_data[0][1].split("：")[1])
		feat_id = pd.DataFrame.from_dict({"id": [subject_id], "name": [name], "label": [{"HC": 0, "WD": 1}[group]],
		                                  "age": [age], "gender_0": [{"male": 0, "female": 1}[gender]],
		                                  "gender_1": [{"male": 1, "female": 0}[gender]]}, orient='index').transpose()
		feat_md, head = None, None
		feat_vowel_n_mean = pd.DataFrame()
		for each_file in os.listdir(subject_dir):
			if each_file == "01_SP":  # 提取SP任务的声学特征
				feat_vowel_n = []
				feat_vowel_n_md = {'VMD': [], 'EMD': [], 'CEEMDAN': []}
				for num in range(1, 4):  # 对每一个元音提取特征，若同一元音存在多个音频，则求均值
					vowel_audio = os.path.join(subject_dir, each_file, subject_id +
					                           "_*.wav".replace("*", str(num)))
					if os.path.exists(vowel_audio):
						if feat == 'MFCC':
							head = ["id", "name", "label", "age", "gender_0", "gender_1", "MFCC"]
							mfcc = calcu_mfcc(vowel_audio)
							feat_vowel_n.append(mfcc)
						else:
							head = ["id", "name", "label", "age", "gender_0", "gender_1",
							        f"{feat}-VMD", f"{feat}-CEEMDAN", f"{feat}-EMD"]
							for md in ['VMD', 'CEEMDAN', 'EMD']:
								decomp_path = os.path.join(self.decomp_dir, group, gender, subject_id, md)
								imfs_name = 'a' + str(num)
								imfs_f = os.path.join(decomp_path, imfs_name + '.npy')
								imfs = np.load(imfs_f)[:self.decomposition_num[md], :]
								if feat == 'DMFCC':
									dmfcc = calcu_decompose_mfcc(imfs, vowel_audio)
									feat_vowel_n_md[md].append(dmfcc)
								else:
									hcc = calcu_hcc(imfs)
									feat_vowel_n_md[md].append(hcc)
				if feat == 'MFCC':
					feat_vowel_n_mean = {'MFCC': [np.mean(np.array(feat_vowel_n), axis=0)]}
				else:
					feat_vowel_n_mean = {f"{feat}-VMD": [np.mean(np.array(feat_vowel_n_md['VMD']), axis=0)],
					                     f"{feat}-CEEMDAN": [np.mean(np.array(feat_vowel_n_md['CEEMDAN']), axis=0)],
					                     f"{feat}-EMD": [np.mean(np.array(feat_vowel_n_md['EMD']), axis=0)]}
		_feat_all = pd.concat([feat_id, pd.DataFrame(feat_vowel_n_mean)], axis=1)
		feat_all = pd.concat([pd.DataFrame(columns=head), _feat_all])  # 防止有的音频不存在导致对应列数据不存在
		feat_all.fillna(np.nan, inplace=True)
		return feat_all

	def run_parallel(self, res_dir: str = '', n_jobs=None):
		"""
		并行处理，保存所有信号分量至本地文件
		:param res_dir: 结果保存路径
		:param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
		:return: None
		"""
		assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
		if n_jobs == -1:
			n_jobs = None
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		fts = ['MFCC', 'DMFCC', 'HCC']
		for ft in fts:
			print(f"---------- Extracting features of {ft} ----------")
			feats_all = pd.DataFrame()
			# for subj in self.subject_dir_list:  # 非并行
			# 	feats_all = pd.concat([self.get_features(subj, ft), feats_all], ignore_index=True)
			_parallel_process = partial(self.get_features, feat=ft)
			with Pool(n_jobs) as pool:
				res = pool.map(_parallel_process, self.subject_dir_list)
			for _res in res:
				feats_all = pd.concat([feats_all, _res], ignore_index=True)
			feats_all.sort_values(by=['id'], inplace=True)
			feats_all.dropna(inplace=True)
			feats_all = feats_all[feats_all['age'] >= 18].reset_index(drop=True)  # 去掉未成年数据
			feats_all.drop_duplicates('name', keep='last', inplace=True, ignore_index=True)  # 去掉重复被试数据，仅保留最近日期的
			feats_all.drop(columns='name', inplace=True)  # 删除姓名列
			feats_all.to_csv(os.path.join(res_dir, f'features_{ft}.csv'), encoding="utf-8-sig", index=False)
			# 序列化存储至本地，保留原始信息
			feats_all.to_pickle(os.path.join(res_dir, f'features_{ft}.pkl'))


def ceemdan_accuracy_plot():
	"""
	获取CEEMDAN对应的IMF数量：通过在默认参数下，SVM/RF/MLP模型对于7:3数据集划分，DMFCC特征，在测试集上准确率最高的IMF数量
	:return: None
	"""
	from matplotlib import pyplot as plt
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.svm import SVC
	from sklearn.neural_network import MLPClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split
	res_path = os.path.join(current_path, r"results")
	hcc_dmfcc = {'HCC-CEEMDAN': [], 'DMFCC-CEEMDAN': []}
	for i_pc in range(1, 11):
		for j_md in hcc_dmfcc.keys():
			data_dir = os.path.normpath(os.path.join(current_path, f"data/features/ceemdan_imf{i_pc}"))
			feat_data = pd.read_pickle(os.path.join(data_dir, 'features_' + j_md.split('-')[0] + '.pkl'))
			_x_data, y_label = feat_data[j_md], feat_data.iloc[:, 1:2]  # 排除性别及年龄特征
			x_data = []
			for i_subj in _x_data:
				x_data.append(i_subj)
			x_data = np.array(x_data)  # shape=[样本数，特征维数]
			train_data, test_data, train_label, test_label = train_test_split(x_data, y_label, random_state=rs,
			                                                                  test_size=0.3, shuffle=True,
			                                                                  stratify=y_label)
			train_label, test_label = np.array(train_label).ravel(), np.array(test_label).ravel()
			# 划分数据后再进行数据处理，避免测试集数据泄露
			ss = StandardScaler()  # 标准化特征
			pipe = Pipeline([('ss', ss)])
			train_data = pipe.fit_transform(train_data)
			test_data = pipe.transform(test_data)
			model_svc = SVC(probability=True, random_state=rs)
			model_cal = CalibratedClassifierCV(model_svc, n_jobs=-1)  # 仅校准最优参数对应的模型
			model_cal.fit(train_data, train_label)
			acc_svc = accuracy_score(test_label, model_cal.predict(test_data))
			model_rf = RandomForestClassifier(n_jobs=-1, random_state=rs)
			model_rf.fit(train_data, train_label)
			acc_rf = accuracy_score(test_label, model_rf.predict(test_data))
			model_mlp = MLPClassifier(max_iter=10000, learning_rate='adaptive', random_state=rs)
			model_mlp.fit(train_data, train_label)
			acc_mlp = accuracy_score(test_label, model_mlp.predict(test_data))
			hcc_dmfcc[j_md].append([acc_svc, acc_rf, acc_mlp])
	plt.figure(figsize=(8, 5))
	plt.title('Accuracy variation on CEEMDAN-based IMFs',
	          fontdict={'family': font_family, 'size': 16})
	plt.xlabel('Number of IMFs', fontdict={'family': font_family, 'size': 14})
	plt.ylabel('Accuracy', fontdict={'family': font_family, 'size': 14})
	# 获取最大索引，当有多个相同最大值时，再比较标准差，取较小的标准差，若当标准差也相同，则取最后一个值
	max_hcc_l = np.max(np.mean(hcc_dmfcc['HCC-CEEMDAN'], axis=1))
	tup = [(i, np.mean(hcc_dmfcc['HCC-CEEMDAN'], axis=1)[i]) for i in range(len(np.mean(hcc_dmfcc['HCC-CEEMDAN'], axis=1)))]
	max_hcc_index_l = [i for i, n in tup if n == max_hcc_l]
	max_hcc_index = max_hcc_index_l[0]
	std_hcc = np.std(hcc_dmfcc['HCC-CEEMDAN'], axis=1)[max_hcc_index]
	if len(max_hcc_index_l) > 1:  # 超过一个最大值
		for i_index in max_hcc_index_l:
			if np.std(hcc_dmfcc['HCC-CEEMDAN'], axis=1)[i_index] <= std_hcc:  # 进一步比较标准差
				max_hcc_index = i_index
				std_hcc = np.std(hcc_dmfcc['HCC-CEEMDAN'], axis=1)[i_index]
	max_hcc = f"HCCs (max = {np.mean(hcc_dmfcc['HCC-CEEMDAN'], axis=1)[max_hcc_index]:.4f}±{std_hcc:.4f}, " \
	          f"IMF {1 + max_hcc_index})"
	max_dmfcc_l = np.max(np.mean(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1))
	tup = [(i, np.mean(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)[i]) for i in range(len(np.mean(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)))]
	max_dmfcc_index_l = [i for i, n in tup if n == max_dmfcc_l]
	max_dmfcc_index = max_dmfcc_index_l[0]
	std_dmfcc = np.std(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)[max_dmfcc_index]
	if len(max_dmfcc_index_l) > 1:
		for i_index in max_dmfcc_index_l:
			if np.std(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)[i_index] <= std_dmfcc:
				max_dmfcc_index = i_index
				std_dmfcc = np.std(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)[i_index]
	max_dmfcc = f"DMFCC (max = {np.mean(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1)[max_dmfcc_index]:.4f}±{std_dmfcc:.4f}, " \
	            f"IMF {1 + max_dmfcc_index})"
	print(f"The max accuracy using HCC: {max_hcc}")
	print(f"The max accuracy using DMFCC: {max_dmfcc}")
	plt.errorbar([i for i in range(1, 11)], np.mean(hcc_dmfcc['HCC-CEEMDAN'], axis=1), elinewidth=1.5,
	             yerr=np.std(hcc_dmfcc['HCC-CEEMDAN'], axis=1), fmt='bo--', ecolor='black', capsize=5,
	             lw=1.2, ms=6, label=max_hcc)
	plt.errorbar([i for i in range(1, 11)], np.mean(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1), elinewidth=1.5,
	             yerr=np.std(hcc_dmfcc['DMFCC-CEEMDAN'], axis=1), fmt='rs-', ecolor='black', capsize=5,
	             lw=1.2, ms=6, label=max_dmfcc)
	plt.legend(loc="upper right", prop={'family': font_family, 'size': 12}, labelspacing=1.0)
	plt.ylim(0.4, 1.0)
	plt.xticks(np.arange(12), [''] + [str(i) for i in range(1, 11)] + [''])
	for sp in plt.gca().spines:
		plt.gca().spines[sp].set_color('black')
		plt.gca().spines[sp].set_linewidth(1)
	plt.gca().tick_params(direction='in', color='black', length=5, width=1)
	plt.grid(False)
	plt.tight_layout()
	plt.savefig(os.path.join(res_path, f"optimal_ceemdan_imf{1 + max_dmfcc_index}.png"), dpi=600)
	plt.savefig(os.path.join(res_path, f"optimal_ceemdan_imf{1 + max_dmfcc_index}.svg"), format='svg')
	plt.show()
	plt.close('all')


if __name__ == "__main__":
	start_time = datetime.datetime.now()
	print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
	current_path = os.getcwd()  # 获取当前文件夹
	audio_data = os.path.join(current_path, 'data/audio')
	decomposition_path = os.path.join(current_path, r'data/decomposition')
	output_path = os.path.join(current_path, r'data/features')

	# ceemdan_accuracy_plot()  # 首先针对CEEMDAN，获取最佳的IMF数量，为8（仅运行一次）

	gf = GetFeatures(audio_data, decomposition_path)
	gf.run_parallel(output_path, n_jobs=-1)

	end_time = datetime.datetime.now()
	print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
	print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
	with open(r"./results/finished.txt", "w") as ff:
		ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
				 f"({os.path.basename(__file__)}) -------------------\r\n")
		ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
				 f"({os.path.basename(__file__)}) -------------------\r\n")
		ff.write(f"------------------ Time Used {end_time - start_time} "
				 f"({os.path.basename(__file__)}) -------------------\r\n")


