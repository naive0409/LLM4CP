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

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0001
epochs = 500
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_loss = 100
loss_alpha_param = 1

time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
try:
    save_path = "Weights/full_shot_fdd/{}/".format(time_stamp)
    save_path = os.getcwd() + r'/' + save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("New folder at:" + save_path)
    save_path = save_path + "clip.pth"
    print('Weights will be stored at:' + save_path)
except BaseException as msg:
    print("Fail to make new folder:" + msg)

writer = SummaryWriter(comment=time_stamp)

train_TDD_r_path = "./Training Dataset/H_U_his_train.mat"
train_TDD_t_path = "./Training Dataset/H_D_pre_train.mat"
key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']

is_U2D = 1
is_few = 0
pred_len = 4
prev_len = 16
dataset_pickle_name = "./code_testing/dataset_{}_{}_{}_{}.pickle".format(is_U2D, is_few, pred_len, prev_len)

train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=is_U2D, is_few=is_few)  # creat data for training
validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=is_U2D)  # creat data for validation

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
        epoch_train_loss, epoch_train_CLIP_loss, epoch_train_NMSE_loss = [], [], []
        epoch_val_loss, epoch_val_CLIP_loss, epoch_val_NMSE_loss =  [], [],[]
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pred_t, prev = Variable(batch[0]).to(device), \
                           Variable(batch[1]).to(device)
            optimizer.zero_grad()  # fixed
            clip_model_loss_output, pred_m = model(prev, None, None, None)

            # compute loss
            NMSE_loss = criterion(pred_m, pred_t)
            CLIP_loss = loss_alpha_param * clip_model_loss_output
            loss = NMSE_loss + CLIP_loss

            # save all losses into a vector for one epoch
            epoch_train_loss.append(loss.item())
            epoch_train_NMSE_loss.append(NMSE_loss.item())
            epoch_train_CLIP_loss.append(CLIP_loss.item())

            loss.backward()
            optimizer.step()

        #       lr_scheduler.step()  # update lr

        # compute the mean value of all losses, as one epoch loss
        t_loss = np.nanmean(np.array(epoch_train_loss))
        t_NMSE_loss = np.nanmean(np.array(epoch_train_NMSE_loss))
        t_CLIP_loss = np.nanmean(np.array(epoch_train_CLIP_loss))

        print('Epoch: {}/{} training loss: {:.7f},NMSE: {:.7f},CLIP: {:.7f}'.format(epoch+1, epochs, t_loss, t_NMSE_loss, t_CLIP_loss))  # print loss for each epoch

        # writer.add_scalar('training loss', t_loss, epoch)
        # writer.add_scalar('training NMSE loss', t_NMSE_loss, epoch)
        # writer.add_scalar('training CLIP loss', t_CLIP_loss, epoch)
        writer.add_scalars('training loss',
                           {
                               'EPOCH': t_loss,
                               'NMSE': t_NMSE_loss,
                               'CLIP': t_CLIP_loss
                            },
                            epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).to(device), \
                               Variable(batch[1]).to(device)
                optimizer.zero_grad()  # fixed
                clip_model_loss_output, pred_m = model(prev, None, None, None)

                # compute loss
                NMSE_loss = criterion(pred_m, pred_t)
                CLIP_loss = loss_alpha_param * clip_model_loss_output
                loss = NMSE_loss + CLIP_loss

                # save all losses into a vector for one epoch
                epoch_val_loss.append(loss.item())
                epoch_val_NMSE_loss.append(NMSE_loss.item())
                epoch_val_CLIP_loss.append(CLIP_loss.item())

            # compute the mean value of all losses, as one epoch loss
            v_loss = np.nanmean(np.array(epoch_val_loss))
            v_NMSE_loss = np.nanmean(np.array(epoch_val_NMSE_loss))
            v_CLIP_loss = np.nanmean(np.array(epoch_val_CLIP_loss))

            print('validate loss: {:.7f},NMSE: {:.7f},CLIP: {:.7f}'.format(v_loss, v_NMSE_loss, v_CLIP_loss))

            # writer.add_scalar('validate loss', v_loss, epoch)
            # writer.add_scalar('validate NMSE loss', v_NMSE_loss, epoch)
            # writer.add_scalar('validate CLIP loss', v_CLIP_loss, epoch)

            writer.add_scalars('validate loss',
                            {
                                'VALIDATE': v_loss,
                                'NMSE': v_NMSE_loss,
                                'CLIP': v_CLIP_loss
                            },
                            epoch)

            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model)


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

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
