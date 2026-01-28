import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random
import scipy.io as sio
import os


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_r, file_path_t, is_train=1, ir=1, SNR=15, is_U2D=0, is_few=0,
                 train_per=0.9, valid_per=0.1,
                 use_dichasus=False):
        super(Dataset_Pro, self).__init__()
        self.SNR = SNR
        self.ir = ir
        self.dichasus = use_dichasus
        if self.dichasus is False:
            H_his = hdf5storage.loadmat(file_path_r)['H_U_his_train']  # v,b,l,k,a,b,c
            if is_U2D:
                H_pre = hdf5storage.loadmat(file_path_t)["H_D_pre_train"]  # v,b,l,k,a,b,c
            else:
                H_pre = hdf5storage.loadmat(file_path_t)["H_U_pre_train"]  # v,b,l,k,a,b,c
            print(H_his.shape, H_pre.shape)

        if self.dichasus:
            # 1. 加载 .pt 文件
            raw_csi = torch.load(file_path_r, map_location='cpu') # 根据需要调整 map_location

            # 2. 获取 CSI 数据
            raw_csi = raw_csi['csi']
            raw_csi = torch.complex(raw_csi[..., 0], raw_csi[..., 1])

            total_samples, num_antennas, freq_bins = raw_csi.shape
            assert freq_bins >= 360 + 48, f"Frequency bins ({freq_bins}) must be at least {360 + 48}."

            # 3. 提取所需频率索引 (96 个)
            # 索引 300-347 (长度 48) 和 360-407 (长度 48)
            idx1_start, idx1_len = 300, 48
            idx2_start, idx2_len = 350, 48
            # ant_index = range(32)
            ant_index = [6, 2, 16, 18, 28, 5, 10, 14]
            num_antennas = len(ant_index)
            selected_freqs_part1 = raw_csi[:, ant_index, idx1_start:idx1_start + idx1_len] # [18602, 8, 48]
            selected_freqs_part2 = raw_csi[:, ant_index, idx2_start:idx2_start + idx2_len] # [18602, 8, 48]
            selected_freqs = torch.cat([selected_freqs_part1, selected_freqs_part2], dim=2) # [18602, 8, 96]
            print(f"Selected freqs shape: {selected_freqs.shape}") # [18602, 8, 96]

            # ================== 【关键新增：瞬时相位归一化 IPN】 ==================
            # 以每时刻、每子载波的“第 0 根天线”为基准相位
            # ref_ant 形状: [18602, 1, 96]
            ref_ant = selected_freqs[:, 0:1, :]
            ref_phase = ref_ant / (torch.abs(ref_ant) + 1e-9)

            # 抹除每一帧的公共相位旋转
            # 此时 selected_freqs[:, 0, :] 会变成实数（相位为0）
            selected_freqs = selected_freqs * torch.conj(ref_phase)

            # ================== 【关键新增：样本能量归一化】 ======================
            # 实测数据不同时刻能量波动极大，必须做归一化
            # 计算每个时间点的平均幅度
            sample_mag = torch.mean(torch.abs(selected_freqs), dim=(1, 2), keepdim=True)
            selected_freqs = selected_freqs / (sample_mag + 1e-9)
            # ====================================================================

            # 4. 沿第一维分片，每片 20 个样本
            split_size = 20
            if total_samples % split_size != 0:
                # 如果总数不能被 20 整除，丢弃最后不足 20 个的样本
                remaining_samples = total_samples % split_size
                print(f"Warning: Discarding last {remaining_samples} samples that don't form a complete chunk of size {split_size}.")
                selected_freqs = selected_freqs[: (total_samples // split_size) * split_size]

            # 使用 view 重塑进行分片
            num_chunks = selected_freqs.size(0) // split_size
            reshaped_csi = selected_freqs.view(num_chunks, split_size, num_antennas, selected_freqs.size(2))
            # reshaped_csi 形状: [v, l, a=8, k, 2] where v = num_chunks, l=20

            # 5. 交换维度以适应后续处理，使其类似 (v, l, k, a)
            # 原始: [v, l, a, k] -> 目标: [v, l, k, a]
            # 使用 permute 来交换 a 和 k 维度
            permuted_csi = reshaped_csi.permute(0, 1, 3, 2) # [v, l, k, a2]

            ref_symbol = permuted_csi[:, 0:1, 0:1, 0:1]
            ref_phase = ref_symbol / (torch.abs(ref_symbol) + 1e-9)

            # 将整个序列(20帧)乘以基准相位的共轭，相当于把起始相位旋转回 0 度
            # 这样网络只需要学习相对变化，而不需要猜测随机的绝对相位
            permuted_csi = permuted_csi * torch.conj(ref_phase)

            # 打乱样本顺序
            total_chunks = permuted_csi.size(0)
            # indices = np.random.permutation(permuted_csi.shape[0])
            # np.save('indices.npy', indices)
            indices = np.load('/mnt/DataDrive164/wr/LLM4CP/Training Dataset/indices.npy')
            permuted_csi = permuted_csi[indices]
            # 6. 根据 is_train 划分 H_his 和 H_pre
            if is_train:
                end_idx = int(train_per * total_chunks)
                H_his = permuted_csi[:end_idx, :16, :48,...]      # [n_train, 16, k, a]
                H_pre = permuted_csi[:end_idx, -4:, -48:,...]    # [n_train, 4, k, a]
            else:
                start_idx = int(train_per * total_chunks)
                end_idx = int((train_per + valid_per) * total_chunks)
                H_his = permuted_csi[start_idx:end_idx, :16, :48,...]      # [n_valid, 16, k, a]
                H_pre = permuted_csi[start_idx:end_idx, -4:, -48:,...]    # [n_valid, 4, k, a]

            # 合并 k 和 (a*real_imag) 维度
            H_his = rearrange(H_his, 'n L k a -> n L (k a)')
            H_pre = rearrange(H_pre, 'n L k a -> n L (k a)')
        else:
            batch = H_pre.shape[1]
            if is_train:
                H_his = H_his[:, :int(train_per * batch), ...]
                H_pre = H_pre[:, :int(train_per * batch), ...]
            else:
                H_his = H_his[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
                H_pre = H_pre[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_his = rearrange(H_his, 'v n L k a b c -> (v n) L (k a b c)')
            H_pre = rearrange(H_pre, 'v n L k a b c -> (v n) L (k a b c)')

        B, prev_len, mul = H_his.shape
        _, pred_len, mul = H_pre.shape
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.seq_len = pred_len + prev_len

        dt_all = np.concatenate((H_his, H_pre), axis=1)
        np.random.shuffle(dt_all)
        H_his = dt_all[:, :prev_len, ...]
        H_pre = dt_all[:, -pred_len:, ...]
        if self.dichasus is False:
            for i in range(B):
                H_his[i, ...] = noise(H_his[i, ...], random.rand() * 15 + 5.0)
                H_pre[i, ...] = noise(H_pre[i, ...], random.rand() * 15 + 5.0)
        std = np.sqrt(np.std(np.abs(H_his) ** 2))
        H_his = H_his / std
        H_pre = H_pre / std
        H_pre = LoadBatch_ofdm(H_pre, num=len(ant_index))
        H_his = LoadBatch_ofdm(H_his, num=len(ant_index))
        if is_few == 1:
            H_pre = H_pre[::10, ...]
            H_his = H_his[::10, ...]
        self.pred = H_pre  # b,16,(48*2)
        self.prev = H_his  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :].float(), \
               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm(H, num=32):
    # H: B,T,mul             [tensor complex]
    # out:B*num,T,mul*2/num  [tensor real]
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) ->(b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out
