"""
@Project ：LLM4CP
@File    ：test.py
@IDE     ：PyCharm
@Author  ：XvanyvLiu
@mail    : xvanyvliu@gmail.com
@Date    ：2024/4/8 17:11
"""
import time
import torch
import numpy as np
from data import LoadBatch_ofdm_1, LoadBatch_ofdm_2, noise, Transform_TDD_FDD
from metrics import NMSELoss, SE_Loss
from einops import rearrange
import hdf5storage
import tqdm
from pvec import pronyvec
from PAD import PAD3
from scipy.io import savemat
import os

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.nn as nn


def load_dichasus_validation_data(file_path, is_U2D=1, train_per=0.8, valid_per=0.1):
    """
    加载dichasus数据集的验证集部分
    类似于data.py中的处理逻辑，但针对验证集（is_train=0）
    """
    # 1. 加载 .pt 文件
    raw_csi = torch.load(file_path, map_location='cpu')

    # 2. 获取 CSI 数据
    raw_csi = raw_csi['csi']
    raw_csi = torch.complex(raw_csi[..., 0], raw_csi[..., 1])

    total_samples, num_antennas, freq_bins = raw_csi.shape
    assert freq_bins >= 360 + 48, f"Frequency bins ({freq_bins}) must be at least {360 + 48}."

    # 3. 提取所需频率索引 (96 个)
    # 索引 300-347 (长度 48) 和 360-407 (长度 48)
    idx1_start, idx1_len = 300, 48
    idx2_start, idx2_len = 350, 48
    ant_index = [6, 2, 16, 18, 28, 5, 10, 14]
    num_antennas = len(ant_index)
    selected_freqs_part1 = raw_csi[:, ant_index, idx1_start:idx1_start + idx1_len]  # [18602, 8, 48]
    selected_freqs_part2 = raw_csi[:, ant_index, idx2_start:idx2_start + idx2_len]  # [18602, 8, 48]
    selected_freqs = torch.cat([selected_freqs_part1, selected_freqs_part2], dim=2)  # [18602, 8, 96]

    # ================== 【关键新增：瞬时相位归一化 IPN】 ==================
    # 以每时刻、每子载波的"第 0 根天线"为基准相位
    ref_ant = selected_freqs[:, 0:1, :]
    ref_phase = ref_ant / (torch.abs(ref_ant) + 1e-9)

    # 抹除每一帧的公共相位旋转
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

    # 5. 交换维度以适应后续处理，使其类似 (v, l, k, a)
    # 原始: [v, l, a, k] -> 目标: [v, l, k, a]
    permuted_csi = reshaped_csi.permute(0, 1, 3, 2)  # [v, l, k, a2]

    ref_symbol = permuted_csi[:, 0:1, 0:1, 0:1]
    ref_phase = ref_symbol / (torch.abs(ref_symbol) + 1e-9)

    # 将整个序列(20帧)乘以基准相位的共轭，相当于把起始相位旋转回 0 度
    permuted_csi = permuted_csi * torch.conj(ref_phase)

    # 打乱样本顺序
    total_chunks = permuted_csi.size(0)
    indices = np.load('/mnt/DataDrive164/wr/LLM4CP/Training Dataset/indices.npy')
    permuted_csi = permuted_csi[indices]

    # 6. 根据验证集划分提取数据 (is_train=0)
    start_idx = int(train_per * total_chunks)
    end_idx = int((train_per + valid_per) * total_chunks)
    H_his = permuted_csi[end_idx:, :16, :48, ...]  # [n_valid, 16, k, a]
    H_pre = permuted_csi[end_idx:, -4:, -48:, ...]  # [n_valid, 4, k, a]

    # 合并 k 和 (a*real_imag) 维度
    H_his = rearrange(H_his, 'n L k a -> n L (k a)')
    H_pre = rearrange(H_pre, 'n L k a -> n L (k a)')

    # 数据标准化
    # std = np.sqrt(np.std(np.abs(H_his) ** 2))
    # std = torch.sqrt(torch.std(torch.abs(H_his) ** 2))
    std = np.sqrt(np.std(np.abs(H_his.cpu().numpy()) ** 2))
    H_his = H_his / std
    H_pre = H_pre / std

    # 转换为实数格式
    H_his = rearrange(H_his, 'n L (k a) -> n a L k', a=num_antennas)
    H_pre = rearrange(H_pre, 'n L (k a) -> n a L k', a=num_antennas)
    # H_his = LoadBatch_ofdm(H_his, num=len(ant_index))
    # H_pre = LoadBatch_ofdm(H_pre, num=len(ant_index))
    H_his = H_his.cpu().numpy()
    H_pre = H_pre.cpu().numpy()

    print(f"Validation H_his shape: {H_his.shape}")
    print(f"Validation H_pre shape: {H_pre.shape}")

    return H_his, H_pre

class KGNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(KGNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)



if __name__ == "__main__":
    # demo
    device = torch.device('cuda:0')
    deepmimo = False
    dichasus = True
    quadriga_validation_as_test = False  # 使用quadriga的验证集作为测试集
    is_U2D = 1
    prev_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_U_his_test.mat"
    pred_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_U_pre_test.mat"
    pred_path_fdd = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_D_pre_test.mat"
    if deepmimo:
        prev_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_ul_pre_test.mat"
        pred_path_fdd = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_dl_pre_test.mat"
    if quadriga_validation_as_test:
        prev_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/GAP864/H_U_his_train.mat"
        pred_path_fdd = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/GAP864/H_D_pre_train.mat"
    if dichasus:
        dichasus_data_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/dichasus-cf02.pt"
    date_ = '20260127_21_56'
    flops_time_analyze = False
    model_path = {
        'himff': '/mnt/DataDrive164/wr/LLM4CP/Weights/full_shot_fdd/{}/clip.pth'.format(date_),
        'gpt': './Weights/full_shot_fdd/U2D_LLM4CP.pth',
        'transformer': '/home/ubuntu/users/wr/dev/Transformer+KGNET/Weights/Transformer/FDD_backup/U2D_Transformer.pth',
        'cnn': './Weights/full_shot_fdd/U2D_cnn.pth',
        'gru': './Weights/full_shot_fdd/U2D_gru.pth',
        'lstm': './Weights/full_shot_fdd/U2D_lstm.pth',
        'rnn': './Weights/full_shot_fdd/U2D_rnn.pth',
        'kgnet': '/home/ubuntu/users/wr/dev/Transformer+KGNET/Weights/KGNET_new/FDD/U2D_KGNET_varlr001.pth'
    }
    # model_test_enable = [ 'himff', 'gpt', 'transformer', 'cnn', 'gru', 'lstm', 'rnn', 'np']
    model_test_enable = ['himff']
    prev_len = 16
    label_len = 12
    pred_len = 4
    K, Nt, Nr, SR = (48, 4, 4, 1)
    print("Total model nums:", len(model_test_enable))
    # load model and test
    criterion = NMSELoss()
    NMSE = [[] for i in model_test_enable]

    if dichasus:
        # 使用dichasus验证集数据加载函数
        print("Loading dichasus validation dataset...")
        test_data_prev_base, test_data_pred_base = load_dichasus_validation_data(dichasus_data_path, is_U2D=is_U2D)
        # dichasus数据已经经过预处理，不需要额外的重排操作
        print(f"Loaded dichasus validation data - H_his shape: {test_data_prev_base.shape}, H_pre shape: {test_data_pred_base.shape}")
    else:
        # 原有的.mat文件加载逻辑
        test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']
        if is_U2D:
            test_data_pred_base = hdf5storage.loadmat(pred_path_fdd)['H_D_pre_test']
        else:
            test_data_pred_base = hdf5storage.loadmat(pred_path)['H_U_pre_test']
        if deepmimo:
            test_data_prev_base = rearrange(test_data_prev_base, 'a b c d e f g -> (a) (e) (f) (g) (b) (c) (d)')
            test_data_pred_base = rearrange(test_data_pred_base, 'a b c d e f g -> (a) (e) (f) (g) (b) (c) (d)')
            print(test_data_prev_base.shape)
            print(test_data_pred_base.shape)
        if quadriga_validation_as_test:
            # copid from data.py
            # test on validation set
            batch = test_data_pred_base.shape[1]
            train_per = 0.9
            valid_per = 0.1
            test_data_prev_base = test_data_prev_base[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
            test_data_pred_base = test_data_pred_base[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
    for i in range(len(model_test_enable)):
        print("---------------------------------------------------------------")
        print("loading ", i + 1, "th model......", model_test_enable[i])
        if model_test_enable[i] not in ['pad', 'pvec', 'np']:
            model = torch.load(model_path[model_test_enable[i]], map_location=device).to(device)
            # for block in model.MmHFF.MmFF_block_list:
            #     if hasattr(block, 'fusion_flag'): # 检查一下以防万一
            #         block.fusion_flag = False
            #         print(f"Modified fusion_flag for block: {block}, new flag: {block.fusion_flag}")
        for snr in [5 * x for x in range(6)]:  # 0,5,...,25
        # for snr in [20]:  # 0,5,...,25
            for speed in range(0, 1):
                test_loss_stack = []
                test_loss_stack_se = []
                test_loss_stack_se0 = []
                if dichasus:
                    # dichasus数据已经经过预处理，直接使用
                    test_data_prev = test_data_prev_base
                    test_data_pred = test_data_pred_base
                    # 添加噪声
                    test_data_prev = noise(test_data_prev, snr)
                    test_data_pred = noise(test_data_pred, snr)
                    # 重新标准化
                    std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
                    test_data_prev = test_data_prev / std
                    test_data_pred = test_data_pred / std
                    lens = test_data_prev.shape[0]
                else:
                    # 原有的数据处理逻辑
                    test_data_prev = test_data_prev_base[[speed], ...]
                    test_data_pred = test_data_pred_base[[speed], ...]
                    test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
                    test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
                    test_data_prev = noise(test_data_prev, snr)
                    test_data_pred = noise(test_data_pred, snr)
                    std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
                    test_data_prev = test_data_prev / std
                    test_data_pred = test_data_pred / std
                    lens, _, _, _ = test_data_prev.shape
                if model_test_enable[i] in ['himff', 'gpt', 'transformer', 'rnn', 'lstm', 'gru', 'cnn', 'np', 'kgnet']:
                    if model_test_enable[i] != 'np':
                        model.eval()
                    prev_data = LoadBatch_ofdm_2(test_data_prev)
                    pred_data = LoadBatch_ofdm_2(test_data_pred)
                    bs = 1 if flops_time_analyze else 64
                    cycle_times = lens // bs
                    pth = 'code_testing/csi_output/{}'.format(date_)
                    try:
                        os.makedirs(pth)
                    except:
                        pass
                    filename = pth + '/{}_{}dB.mat'.format((speed+1)*10, snr)
                    prev_list = []
                    ground_truth = []
                    model_outputs = []
                    with torch.no_grad():
                        for cyt in range(cycle_times):
                            prev = prev_data[cyt * bs:(cyt + 1) * bs, :, :].to(device)
                            pred = pred_data[cyt * bs:(cyt + 1) * bs, :, :].to(device)
                            prev = rearrange(prev, 'b m l k -> (b m) l k')
                            pred = rearrange(pred, 'b m l k -> (b m) l k')
                            if model_test_enable[i] == 'gpt':
                                out = model(prev, None, None, None)
                            elif model_test_enable[i] == 'transformer':
                                encoder_input = prev
                                dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                                decoder_input = torch.cat([encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                                        dim=1)
                                out = model(encoder_input, decoder_input)
                            elif model_test_enable[i] in ['lstm', 'rnn', 'gru']:
                                out = model(prev, pred_len, device)
                            elif model_test_enable[i] == 'cnn':
                                out = model(prev)
                            elif model_test_enable[i] == 'np':
                                out = prev[:, [-1], :].repeat([1, pred_len, 1])
                            elif model_test_enable[i] == 'himff':
                                '''
                                计算Flops和推理时间：
                                    使用fvcore.nn.FlopCountAnalysis
                                    需要更改models/GPT4CP.py,将model的forward()改为只接受一个参数,即out = model(prev)
                                    需要更改bs = 1
                                    flops.total()可能需要除以16
                                '''
                                if flops_time_analyze:
                                    print(prev.shape)
                                    flops = FlopCountAnalysis(model, torch.randn(16, 16, 96).to(device))
                                    print(flops)
                                    print("FLOPs(G): ", flops.total()/1e9)
                                    time_1 = time.time_ns()
                                # out = model(prev, None, None, None)
                                out = model(prev)
                                if flops_time_analyze:
                                    infer_time = time.time_ns() - time_1
                                    print(out.shape)
                                    print(f"infer time:{infer_time/1e6:.2f} ms.")
                                    assert 0==1
                            elif model_test_enable[i] == 'kgnet':
                                out = model(prev)
                                out = out[:, -4:, :]

                            loss = criterion(out, pred)
                            test_loss_stack.append(loss.item())
                            prev_list.append(prev.cpu().detach().numpy())
                            ground_truth.append(pred.cpu().detach().numpy())
                            model_outputs.append(out.cpu().detach().numpy())
                    savemat(filename, {'ground_truth':np.array(ground_truth),
                                       'model_output':np.array(model_outputs),
                                    #    'prev':np.array(prev_list)
                                       })
                    print("speed:", (speed + 1) * 10, "snr:", snr, "\t: NMSE:", np.nanmean(np.array(test_loss_stack)))
                    NMSE[i].append(np.nanmean(np.array(test_loss_stack)))

        fout_nmse = open(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_{}_data_nmse_fdd_full.csv".format(date_), "w")
        for row in NMSE:
            row = list(map(str, row))
            fout_nmse.write(','.join(row))
            fout_nmse.write('\n')
        fout_nmse.close()

