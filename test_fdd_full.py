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

if __name__ == "__main__":
    # demo
    device = torch.device('cuda:1')
    is_U2D = 1
    prev_path = "./Testing Dataset/H_U_his_test.mat"
    pred_path = "./Testing Dataset/H_U_pre_test.mat"
    pred_path_fdd = "./Testing Dataset/H_D_pre_test.mat"
    model_path = {
        'clip': 'Weights/full_shot_fdd/20241116_17_42/clip.copy.pth',
        'gpt': './Weights/full_shot_fdd/U2D_LLM4CP.pth',
        'transformer': './Weights/full_shot_fdd/U2D_trans.pth',
        'cnn': './Weights/full_shot_fdd/U2D_cnn.pth',
        'gru': './Weights/full_shot_fdd/U2D_gru.pth',
        'lstm': './Weights/full_shot_fdd/U2D_lstm.pth',
        'rnn': './Weights/full_shot_fdd/U2D_rnn.pth'
    }
    # model_test_enable = ['gpt', 'transformer', 'cnn', 'gru', 'lstm', 'rnn', 'np']
    model_test_enable = ['clip']
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
    for i in range(len(model_test_enable)):
        print("---------------------------------------------------------------")
        print("loading ", i + 1, "th model......", model_test_enable[i])
        if model_test_enable[i] not in ['pad', 'pvec', 'np']:
            model = torch.load(model_path[model_test_enable[i]], map_location=device).to(device)
        for speed in range(0, 10):
            test_loss_stack = []
            test_loss_stack_se = []
            test_loss_stack_se0 = []
            test_data_prev = test_data_prev_base[[speed], ...]
            test_data_pred = test_data_pred_base[[speed], ...]
            test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
            std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
            test_data_prev = test_data_prev / std
            test_data_pred = test_data_pred / std
            lens, _, _, _ = test_data_prev.shape
            if model_test_enable[i] in ['clip', 'gpt', 'transformer', 'rnn', 'lstm', 'gru', 'cnn', 'np']:
                if model_test_enable[i] != 'np':
                    model.eval()
                prev_data = LoadBatch_ofdm_2(test_data_prev)
                pred_data = LoadBatch_ofdm_2(test_data_pred)
                bs = 64
                cycle_times = lens // bs
                filename = 'code_testing/csi_output/20241116_17_42/' + '{}.mat'.format((speed+1)*10)
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
                        elif model_test_enable[i] == 'clip':
                            clip_loss, out = model(prev, None, None, None)
                        loss = criterion(out, pred)
                        test_loss_stack.append(loss.item())
                        ground_truth.append(pred.cpu().detach().numpy())
                        model_outputs.append(out.cpu().detach().numpy())
                savemat(filename, {'ground_truth':np.array(ground_truth),'model_output':np.array(model_outputs)})
                print("speed", (speed + 1) * 10, ":  NMSE:", np.nanmean(np.array(test_loss_stack)))
                NMSE[i].append(np.nanmean(np.array(test_loss_stack)))

        fout_nmse = open(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_data_nmse_tdd_full.csv", "w")
        for row in NMSE:
            row = list(map(str, row))
            fout_nmse.write(','.join(row))
            fout_nmse.write('\n')
        fout_nmse.close()

