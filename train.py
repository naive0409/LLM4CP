import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP import Model
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from metrics import NMSELoss, SE_Loss
import pickle
import datetime
from torchsummary import summary


# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.00001
epochs = 500
batch_size = 512
dichasus = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_loss = 100
loss_alpha_param = 1

is_U2D = 1
is_few = 0
pred_len = 4
prev_len = 16

time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
try:
    if is_U2D == 1:
        save_path = "/mnt/DataDrive164/wr/LLM4CP/Weights/full_shot_fdd/{}/".format(time_stamp)
    else:
        save_path = "/mnt/DataDrive164/wr/LLM4CP/Weights/full_shot_tdd/{}/".format(time_stamp)
    # save_path = os.getcwd() + r'/' + save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("New folder at:" + save_path)
    save_path = save_path + "clip.pth"
    print('Weights will be stored at:' + save_path)
except BaseException as msg:
    print("Fail to make new folder:" + msg)

writer = SummaryWriter(comment=time_stamp)

train_TDD_r_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/default/H_U_his_train.mat"
if is_U2D == 1:
    train_TDD_t_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/default/H_D_pre_train.mat"
else:
    train_TDD_t_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/default/H_U_pre_train.mat"

if dichasus == True:
    train_TDD_r_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/dichasus-cf02.pt"
    if is_U2D == 1:
        train_TDD_t_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/dichasus-cf0x/H_D_pre_train.mat"
    else:
        train_TDD_t_path = "/mnt/DataDrive164/wr/LLM4CP/Training Dataset/dichasus-cf0x/H_U_pre_train.mat"
key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']

dataset_pickle_name = "./code_testing/dataset_{}_{}_{}_{}.pickle".format(is_U2D, is_few, pred_len, prev_len)

train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=is_U2D, is_few=is_few,
                        use_dichasus=dichasus)  # creat data for training
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=is_U2D,
                           use_dichasus=dichasus)  # creat data for validation

# with open(dataset_pickle_name, "wb") as f:
#     pickle.dump(train_set, f)
#     pickle.dump(validate_set, f)

# with open(dataset_pickle_name, "rb") as f:
#     train_set = pickle.load(f)
#     validate_set = pickle.load(f)

model = Model(pred_len=pred_len, prev_len=prev_len,
              UQh=1, UQv=1, BQh=1, BQv=1).to(device)
if os.path.exists(save_path):
    model = torch.load(save_path, map_location=device)


def save_best_checkpoint(model):  # save model function
    model_out_path = save_path
    torch.save(model, model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(training_data_loader, validate_data_loader):
    global epochs, best_loss
    print('Start training...')
    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_val_loss =  []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pred_t, prev = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device)
            optimizer.zero_grad()  # fixed
            # pred_m = model(prev, None, None, None)
            pred_m = model(prev)

            # compute loss
            NMSE_loss = criterion(pred_m, pred_t)
            loss = NMSE_loss

            # save all losses into a vector for one epoch
            epoch_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        #       lr_scheduler.step()  # update lr

        # compute the mean value of all losses, as one epoch loss
        t_loss = np.nanmean(np.array(epoch_train_loss))

        print('Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, epochs, t_loss))  # print loss for each epoch

        writer.add_scalars('training loss', {'EPOCH': t_loss,}, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).to(device), \
                               Variable(batch[1]).to(device)
                optimizer.zero_grad()  # fixed
                # pred_m = model(prev, None, None, None)
                pred_m = model(prev)

                # compute loss
                NMSE_loss = criterion(pred_m, pred_t)
                loss = NMSE_loss

                # save all losses into a vector for one epoch
                epoch_val_loss.append(loss.item())

            # compute the mean value of all losses, as one epoch loss
            v_loss = np.nanmean(np.array(epoch_val_loss))

            print('validate loss: {:.7f}'.format(v_loss))

            writer.add_scalars('validate loss', {'VALIDATE': v_loss,}, epoch)

            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model)


def print_model_structure(model, indent=0, file=None):
    """递归打印模型结构并写入文件"""
    for name, module in model.named_children():
        # 获取当前层的直接参数（不递归子层）
        params = list(module.parameters(recurse=False))
        num_params = sum(p.numel() for p in params)

        # 格式化参数量和冻结状态
        params_str = ""
        if num_params > 0:
            num_params_m = num_params / 1e6
            params_str = f", Params: {num_params_m:.6f}M"
            # 判断是否所有参数均被冻结
            is_frozen = all(not p.requires_grad for p in params)
            if is_frozen:
                params_str += " (Frozen)"

        # 构建输出行
        line = ' ' * indent + f"({name}): {module.__class__.__name__}{params_str}"
        # print(line)  # 控制台输出
        if file:
            file.write(line + '\n')  # 写入文件

        # 递归遍历子层
        if list(module.children()):
            print_model_structure(module, indent + 4, file)

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    with open("./code_testing/structure_model.txt", "w") as f:
        print_model_structure(model, file=f)

    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion = NMSELoss().to(device)
    train(training_data_loader, validate_data_loader)  # call train function (

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
