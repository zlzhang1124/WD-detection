#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/12/23 15:49 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : signal_decomposition.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V2.0 - ZL.Z：2022/2/4
#             V1.0 - ZL.Z：2021/12/23 - 2021/12/27
# 		      First version.
# @License  : None
# @Brief    : 语音信号分解：分别利用EMD/CEEMDAN/VMD

from config import *
import io
import datetime
import psutil
import glob
import matplotlib.pyplot as plt
import librosa
import parselmouth
from parselmouth.praat import call
from pathos.pools import ProcessPool as Pool
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import PyEMD
import vmdpy
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# import time


class SignalDecomposition:
	"""语音信号分解"""
	def __init__(self):
		"""
		初始化
		:return: None
		"""
		self.audio_f_list = []  # 每个被试的所有音频文件组成的列表

	@staticmethod
	def get_decomposition(signal: np.ndarray, method: str = 'CEEMDAN',
						  show=False, point_scale=1.0, save_fig=None):
		"""
		对信号进行类经验模态分解EMD
		:param signal: 输入信号
		:param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
		:param show: 是否对分解后的信号分量（本征模态分量IMFs或乘积函数PF）进行显示绘制
		:param point_scale: 信号的横轴时间点尺度，即每个点对应的毫秒数
		:param save_fig: 保存路径，默认为None，不保存
		:return: 分解后的信号分量：np.ndarray[shape=(信号分量数, 采样点数), dtype=float64]
		"""
		assert method in ['EMD', 'CEEMDAN', 'VMD'], "method仅接受'EMD'/'CEEMDAN'/'VMD'输入"
		signal = np.array(signal)
		if method == 'CEEMDAN':
			emd_obj = PyEMD.EMD(DTYPE=np.float16, spline_kind='linear')
			ceemdan = PyEMD.CEEMDAN(trials=50, ext_EMD=emd_obj,
									parallel=True, processes=max(psutil.cpu_count(logical=False) // 6, 1))
			ceemdan.noise_seed(rs)
			with warnings.catch_warnings():  # 防止EMD.py:748: RuntimeWarning: divide by zero encountered in true_divide
				warnings.simplefilter('ignore', RuntimeWarning)
				c_imfs = ceemdan.ceemdan(signal, max_imf=11)
			c_residual = ceemdan.residue
			imfs_residual = np.vstack((c_imfs, c_residual))
		elif method == 'VMD':
			imfs_residual, u_hat, cen_fs = vmdpy.VMD(signal, alpha=2000, tau=0, K=8, DC=0,
													 init=1, tol=1e-6)
		else:  # EMD, range_thr=0.005, total_power_thr=0.01
			emd = PyEMD.EMD(DTYPE=np.float16, spline_kind='quadratic')
			with warnings.catch_warnings():  # 防止EMD.py:748: RuntimeWarning: divide by zero encountered in true_divide
				warnings.simplefilter('ignore', RuntimeWarning)
				imfs_residual = emd.emd(signal, max_imf=16)
		if show:
			mms = MinMaxScaler((-1, 1))
			mms_signal = mms.fit_transform(signal.reshape(-1, 1))
			imf_res_num = imfs_residual.shape[0]
			fig, ax = plt.subplots(imf_res_num + 1, 1, sharex='col', figsize=(7, 9))
			ax[0].plot([i * point_scale for i in range(len(signal))], mms_signal.reshape(len(signal), -1), 'r')
			ax[0].tick_params(labelsize=8)
			ax[0].set_ylabel("Signal", fontdict={'fontsize': 9})
			ax[0].set_ylim(-1.0, 1.0)
			ylabel = 'IMF '
			for i in range(imf_res_num):
				mms_imfs_residual = mms.transform(imfs_residual[i].reshape(-1, 1))
				ax[i + 1].plot([i * point_scale for i in range(len(imfs_residual[i]))],
				               mms_imfs_residual.reshape(len(imfs_residual[i]), -1), 'g', lw=1.0)
				ax[i + 1].tick_params(labelsize=8)
				if i < imf_res_num - 1:
					ax[i + 1].set_ylabel(ylabel + str(i + 1), fontdict={'fontsize': 9})
				else:
					ax[i + 1].set_ylabel("Residual", fontdict={'fontsize': 9})
				ax[i + 1].set_ylim(-1.0, 1.0)
			ax[-1].set_xlabel("Time (ms)", fontdict={'fontsize': 12})
			fig.align_ylabels()
			fig.text(0.03, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical', fontdict={'fontsize': 12})
			plt.tight_layout(rect=(0.05, 0, 1, 1))
			if save_fig is not None:
				plt.savefig(save_fig, dpi=600, bbox_inches='tight', pad_inches=0.2)
			# plt.show()
			plt.close()
		return imfs_residual

	def decomposition_save(self, audio_f: str = '', res_dir: str = '', method: str = 'CEEMDAN'):
		"""
		提取全部信号分量，并保存为本地npy文件
		:param audio_f: 被试主文件夹路径
		:param res_dir: 结果保存路径
		:param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
		:return: None
		"""
		try:
			print("---------- Processing %d / %d: ./%s ----------" %
				  (self.audio_f_list.index(audio_f) + 1, len(self.audio_f_list),
				   os.path.relpath(audio_f, os.getcwd())))
		except ValueError:
			pass
		group = os.path.normpath(audio_f).split(os.sep)[-5]
		gender = os.path.normpath(audio_f).split(os.sep)[-4]
		subject_id = os.path.normpath(audio_f).split(os.sep)[-1].split("_")[0]
		save_path = os.path.join(res_dir, group, gender, subject_id, method)
		if not os.path.exists(save_path):
			os.makedirs(save_path, exist_ok=True)
		signal, sr = librosa.load(audio_f, sr=None)
		save_name = 'a' + os.path.normpath(audio_f).split(os.sep)[-1][-5]
		save_f = os.path.join(save_path, save_name + '.npy')
		# aa = int(time.strftime("%d", time.localtime(os.path.getmtime(save_f))))
		if not os.path.exists(save_f):
			imfs_residual = self.get_decomposition(signal, method)
			np.save(save_f, imfs_residual)
			print(f'{subject_id}_{save_name} npy file saved ({method})')
		else:
			print(f'{subject_id}_{save_name} npy file already exists, skipped ({method})')
		if not os.path.exists(save_f.replace('.npy', '.png')):
			self.get_decomposition(signal[len(signal) // 2 - 800: len(signal) // 2 + 800], method,
								   True, 1000 / sr, save_fig=save_f.replace('.npy', '.png'))
			print(f'{subject_id}_{save_name} png file saved ({method})')
		else:
			print(f'{subject_id}_{save_name} png file already exists, skipped ({method})')

	def run_parallel(self, data_dir: str = '', res_dir: str = '', method: str = 'CEEMDAN', n_jobs=None):
		"""
		并行处理，保存所有信号分量至本地文件
		:param data_dir: 数据文件路径
		:param res_dir: 结果保存路径
		:param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
		:param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
		:return: None
		"""
		assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
		if n_jobs == -1:
			n_jobs = None
		for wav_file in sorted(glob.glob(os.path.join(data_dir, r"**/*_[1-3].wav"), recursive=True)):
			self.audio_f_list.append(wav_file)
		_parallel_process = partial(self.decomposition_save, res_dir=res_dir, method=method)
		with Pool(n_jobs) as pool:
			pool.map(_parallel_process, self.audio_f_list)


def lpc_spectrum_decomposition(klatt=False):
	"""
	对使用Klatt合成器合成的元音/a/进行LPC分析，用于根据共振峰信息进行IMF数量的确定
	:param: klatt: 是否利用Klatt合成器合成元音,并进行LPC分析
	:return: None
	"""
	f0, f1, f2, f3, f4 = 125, 800, 1200, 2300, 2800
	audio_syn = os.path.join(data_synthesized, 'KlattSynthesizer_a.wav')
	if klatt:
		if not os.path.exists(data_synthesized):
			os.makedirs(data_synthesized, exist_ok=True)
		kg = call("Create KlattGrid from vowel", "a", 0.4, f0, f1, 50, f2, 50, f3, 100, f4, 0.05, 1000)
		call(kg, 'To Sound').save(audio_syn, 'WAV')
		audio_syn_sig, audio_syn_sr = librosa.load(audio_syn, sr=None)
		# 对合成语音进行信号分解,以便后续根据LP分析共振峰信息进行IMF数量的确定
		klatt_sd = SignalDecomposition()
		for i_md in mds.keys():
			klatt_save_syn = os.path.join(data_synthesized, i_md + '.npy')
			if not os.path.exists(klatt_save_syn):
				klatt_syn_imfs_residual = klatt_sd.get_decomposition(audio_syn_sig, i_md)
				np.save(klatt_save_syn, klatt_syn_imfs_residual)
				print(f'npy file saved ({i_md})')
			else:
				print(f'npy file already exists, skipped ({i_md})')
	lpc_spectrum_dir = os.path.join(data_synthesized, 'lpc_spectrum_fig')
	if not os.path.exists(lpc_spectrum_dir):
		os.makedirs(lpc_spectrum_dir, exist_ok=True)
	sound = parselmouth.Sound(audio_syn)
	formant_obj = call(sound, "To Formant (burg)", 0, 6, 5000, 0.025, 50.0)
	lpc_obj = call(formant_obj, "To LPC", 16000.0)
	lpc_spectrum_obj = call(lpc_obj, "To Spectrum (slice)", 0.0, 20.0, 0.0, 50.0)
	lpc_tabulate_obj = call(lpc_spectrum_obj, "Tabulate", 'no', 'yes', 'no', 'no', 'no', 'yes')
	table_string = parselmouth.praat.call(lpc_tabulate_obj, "List", 'no')
	df = pd.read_csv(io.StringIO(table_string), sep='\t')
	power_density = np.array(df['pow(dB/Hz)'])
	frequencies = np.array(df['freq(Hz)'])
	mds_dict = {'VMD': os.path.join(data_synthesized, 'VMD.npy'),
	            'EMD': os.path.join(data_synthesized, 'EMD.npy'), 'CEEMDAN': os.path.join(data_synthesized, 'CEEMDAN.npy'), }
	for md_name in mds_dict.keys():
		plt.figure(figsize=(8, 5))
		plt.title(f'LPC Spectrum of Signal Decomposition ({md_name})', fontdict={'family': font_family, 'fontsize': 16})
		plt.xlabel('Frequency (Hz)', fontdict={'family': font_family, 'fontsize': 14})
		plt.ylabel('Sound Pressure Level (dB/Hz)', fontdict={'family': font_family, 'fontsize': 14})
		plt.xticks(fontproperties=font_family, size=12)
		plt.yticks(fontproperties=font_family, size=12)
		plt.plot(frequencies, power_density, c='r', lw=2.0, label='Speech', marker='*', markevery=7)
		imf_param = {'IMF 1': ['blue', 's'], 'IMF 2': ['aqua', 'o'], 'IMF 3': ['darkorange', '^'],
		             'IMF 4': ['green', 'v'], 'IMF 5': ['gray', 'p'], 'IMF 6': ['black', 'x'],
		             'IMF 7': ['deeppink', 'd'], 'IMF 8': ['darkviolet', '|']}
		imfs = np.load(mds_dict[md_name])[:8, :]  # 仅分析前8个分量
		for i_imf in range(len(imfs)):
			formant_obj = call(parselmouth.Sound(imfs[i_imf, :]), "To Formant (burg)", 0, 6, 5000, 0.025, 50.0)
			lpc_obj = call(formant_obj, "To LPC", 16000.0)
			lpc_spectrum_obj = call(lpc_obj, "To Spectrum (slice)", 0.0, 20.0, 0.0, 50.0)
			lpc_tabulate_obj = call(lpc_spectrum_obj, "Tabulate", 'no', 'yes', 'no', 'no', 'no', 'yes')
			table_string = parselmouth.praat.call(lpc_tabulate_obj, "List", 'no')
			df = pd.read_csv(io.StringIO(table_string), sep='\t')
			plt.plot(np.array(df['freq(Hz)']), np.array(df['pow(dB/Hz)']), c=list(imf_param.values())[i_imf][0],
			         lw=1.5, label=list(imf_param.keys())[i_imf], marker=None, markevery=30)
		plt.legend(prop={'family': font_family, 'size': 12}, loc="upper right", labelspacing=0.5)
		plt.xlim(0, 4000)
		plt.ylim(0)
		bottom, top = plt.ylim()
		freq_dict = {f1: 'F1', f2: 'F2', f3: 'F3', f4: 'F4'}
		for freq in freq_dict.keys():
			plt.axvline(x=freq, ymax=0.97, c='r', ls=":", lw=1.5)
			plt.annotate(freq_dict[freq], xy=(freq, top - 5), xytext=(freq-300, top - 15),
			             size=12, arrowprops=dict(arrowstyle="->"))
		plt.tight_layout()
		plt.savefig(os.path.join(lpc_spectrum_dir, md_name + '.png'), dpi=600)
		plt.show()
		plt.close()


if __name__ == "__main__":
	start_time = datetime.datetime.now()
	print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
	current_path = os.getcwd()  # 获取当前文件夹
	audio_data = os.path.join(current_path, 'data/audio')
	data_synthesized = os.path.join(current_path, 'data/synthesized_utterance')
	output_path = os.path.join(current_path, r'data/decomposition')
	mds = {'VMD': 1, 'EMD': 40, 'CEEMDAN': 40, }

	lpc_spectrum_decomposition()  # 对合成语音进行信号分解,以便后续根据LP分析共振峰信息进行IMF数量的确定
	for md in mds.keys():  # 获取EMD信号分量
		print(f"---------- Begin {md} ----------")
		sd = SignalDecomposition()
		sd.run_parallel(audio_data, output_path, md, n_jobs=mds[md])

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
