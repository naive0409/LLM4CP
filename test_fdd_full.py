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
    is_U2D = 1
    prev_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_U_his_test.mat"
    pred_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_U_pre_test.mat"
    pred_path_fdd = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_D_pre_test.mat"
    if deepmimo:
        prev_path = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_ul_pre_test.mat"
        pred_path_fdd = "/mnt/DataDrive164/wr/LLM4CP/Testing Dataset/H_dl_pre_test.mat"
    date_ = '20260104_11_46'
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
    for i in range(len(model_test_enable)):
        print("---------------------------------------------------------------")
        print("loading ", i + 1, "th model......", model_test_enable[i])
        if model_test_enable[i] not in ['pad', 'pvec', 'np']:
            model = torch.load(model_path[model_test_enable[i]], map_location=device).to(device)
            # for block in model.MmHFF.MmFF_block_list:
            #     if hasattr(block, 'fusion_flag'): # 检查一下以防万一
            #         block.fusion_flag = False
            #         print(f"Modified fusion_flag for block: {block}, new flag: {block.fusion_flag}")
        # for snr in [5 * x for x in range(6)]:  # 0,5,...,25
        for snr in [20]:  # 0,5,...,25
            for speed in range(0, 10):
                test_loss_stack = []
                test_loss_stack_se = []
                test_loss_stack_se0 = []
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
                            ground_truth.append(pred.cpu().detach().numpy())
                            model_outputs.append(out.cpu().detach().numpy())
                    savemat(filename, {'ground_truth':np.array(ground_truth),'model_output':np.array(model_outputs)})
                    print("speed:", (speed + 1) * 10, "snr:", snr, "\t: NMSE:", np.nanmean(np.array(test_loss_stack)))
                    NMSE[i].append(np.nanmean(np.array(test_loss_stack)))

        fout_nmse = open(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_{}_data_nmse_fdd_full.csv".format(date_), "w")
        for row in NMSE:
            row = list(map(str, row))
            fout_nmse.write(','.join(row))
            fout_nmse.write('\n')
        fout_nmse.close()

