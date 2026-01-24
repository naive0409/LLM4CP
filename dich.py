#!/usr/bin/env python3
import tensorflow as tf
import scipy.io as sio
import numpy as np

raw_dataset = tf.data.TFRecordDataset(
    [
        # _:02-04; 2:02-06, 3:02
        "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf02.tfrecords",
        "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf03.tfrecords",
        "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf04.tfrecords",
        # "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf05.tfrecords",
        # "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf06.tfrecords",
        # "/mnt/DataDrive164/wr/DICHASUS/cf0x/dichasus-cf07.tfrecords",
     ]
)

feature_description = {
    "cfo": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "snr": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
}


def record_parse_function(proto):
    record = tf.io.parse_single_example(proto, feature_description)

    # Measured carrier frequency offset between MOBTX and each receive antenna.
    cfo = tf.ensure_shape(tf.io.parse_tensor(record["cfo"], out_type=tf.float32), (32))

    # Channel coefficients for all antennas, over all subcarriers, real and imaginary parts
    csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (32, 1024, 2))

    # Time in seconds to closest known tachymeter position. Indicates quality of linear interpolation.
    gt_interp_age_tachy = tf.ensure_shape(record["gt-interp-age-tachy"], ())

    # Position of transmitter determined by a tachymeter pointed at a prism mounted on top of the antenna, in meters (X / Y / Z coordinates)
    pos_tachy = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))

    # Signal-to-Noise ratio estimates for all antennas
    snr = tf.ensure_shape(tf.io.parse_tensor(record["snr"], out_type=tf.float32), (32))

    # Timestamp since start of measurement campaign, in seconds
    time = tf.ensure_shape(record["time"], ())

    return cfo, csi, gt_interp_age_tachy, pos_tachy, snr, time


dataset = raw_dataset.map(record_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Optional: Cache dataset in RAM for faster training
dataset = dataset.cache()

print('Cache dataset done')

# 读取数据并转换为numpy数组
# all_cfo = []
csi_array = []
# all_gt_interp_age_tachy = []
all_pos_tachy = []
all_snr = []
all_time = []

for cfo, csi, gt_interp_age_tachy, pos_tachy, snr, time in dataset:
    # all_cfo.append(cfo.numpy())
    csi_array.append(csi.numpy())
    # all_gt_interp_age_tachy.append(gt_interp_age_tachy.numpy())
    all_pos_tachy.append(pos_tachy.numpy())
    all_snr.append(snr.numpy())
    all_time.append(time.numpy())

# 转换为numpy数组
# cfo_array = np.array(all_cfo)
csi_array = np.array(csi_array)
# gt_interp_age_tachy_array = np.array(all_gt_interp_age_tachy)
pos_tachy_array = np.array(all_pos_tachy)
snr_array = np.array(all_snr)
time_array = np.array(all_time)

print(f"csi_array形状: {csi_array.shape}")
print(f"pos_tachy_array形状: {pos_tachy_array.shape}")
print(f"snr_array形状: {snr_array.shape}")
print(f"time_array形状: {time_array.shape}")

indices = np.concatenate([np.arange(513-48,513), np.arange(513,513+48)])
csi_array = csi_array[:, :, indices, :]

# 步骤2: 按20个一组分组，不足丢弃
group_size = 20
num_complete_groups = len(csi_array) // group_size
csi_array = csi_array[:num_complete_groups * group_size]  # [x*20, 32, 96, 2]
pos_tachy_array = pos_tachy_array[:num_complete_groups * group_size,:]
snr_array = snr_array[:num_complete_groups * group_size,:]
time_array = time_array[:num_complete_groups * group_size]

# 步骤3: 重塑为 [x, 20, 96, 32, 2]
csi_array = csi_array.reshape(num_complete_groups, group_size, 32, 96, 2)
csi_array = np.transpose(csi_array, (0, 1, 3, 2, 4))  # [x, 20, 96, 32, 2]
pos_tachy_array = pos_tachy_array.reshape(num_complete_groups, group_size, 3)
snr_array = snr_array.reshape(num_complete_groups, group_size, 32)
time_array = time_array.reshape(num_complete_groups, group_size)
print(f"csi reshaped:{csi_array.shape}")  # [x, 20, 96, 32, 2]
print(f"pos tachy reshaped:{pos_tachy_array.shape}")  # (930, 20, 3)
print(f"snr reshaped:{snr_array.shape}")  # (930, 20, 32)
print(f"time reshaped:{time_array.shape}")  # (930, 20)


# 步骤4: 转换为复数矩阵
csi_array = csi_array[..., 0] + 1j * csi_array[..., 1]  # [x, 20, 96, 32]

csi_ul = csi_array[:, :16, :48, :]
csi_ul = csi_ul.reshape(num_complete_groups, 16, 48 * 32) # [x,16, subcarrier48*ant32]
csi_dl = csi_array[:, -4:, -48:, :]
csi_dl = csi_dl.reshape(num_complete_groups, 4, 48 * 32)  # [x,16, subcarrier48*ant32]

# 保存为.mat文件
H_U_his_train = {
    # 'cfo': cfo_array,
    'H_U_his_train': csi_ul,
    # 'gt_interp_age_tachy': gt_interp_age_tachy_array,
    'pos_tachy': pos_tachy_array,
    'snr': snr_array,
    'time': time_array
}

H_D_pre_train = {
    'H_D_pre_train': csi_dl,
    'pos_tachy': pos_tachy_array,
    'snr': snr_array,
    'time': time_array
}

sio.savemat('H_U_his_train3.mat', H_U_his_train)
print(f"成功保存数据到 H_U_his_train.mat")
sio.savemat('H_D_pre_train3.mat', H_D_pre_train)
print(f"成功保存数据到 H_D_pre_train.mat")
print(f"数据形状:")
# print(f"CFO: {cfo_array.shape}")
print(f"UL: {csi_ul.shape}")  # (930, 16, 1536)
print(f"DL: {csi_dl.shape}")  # (930, 4, 1536)
# print(f"Position: {pos_tachy_array.shape}")

pass
