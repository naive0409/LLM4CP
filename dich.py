#!/usr/bin/env python3
import tensorflow as tf
import scipy.io as sio
import numpy as np

raw_dataset = tf.data.TFRecordDataset(
    # ["tfrecords/dichasus-cf02.tfrecords", "tfrecords/dichasus-cf03.tfrecords", "tfrecords/dichasus-cf04.tfrecords",
    #  "tfrecords/dichasus-cf05.tfrecords", "tfrecords/dichasus-cf06.tfrecords", "tfrecords/dichasus-cf07.tfrecords"]
    ["./Training Dataset/DICHASUS/cf0x/dichasus-cf02.tfrecords"]
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

print('done')

# 读取数据并转换为numpy数组
all_cfo = []
all_csi = []
all_gt_interp_age_tachy = []
all_pos_tachy = []
all_snr = []
all_time = []

for cfo, csi, gt_interp_age_tachy, pos_tachy, snr, time in dataset:
    # all_cfo.append(cfo.numpy())
    all_csi.append(csi.numpy())
    # all_gt_interp_age_tachy.append(gt_interp_age_tachy.numpy())
    # all_pos_tachy.append(pos_tachy.numpy())
    # all_snr.append(snr.numpy())
    # all_time.append(time.numpy())

# 转换为numpy数组
# cfo_array = np.array(all_cfo)
csi_array = np.array(all_csi)
# gt_interp_age_tachy_array = np.array(all_gt_interp_age_tachy)
# pos_tachy_array = np.array(all_pos_tachy)
# snr_array = np.array(all_snr)
# time_array = np.array(all_time)

'''
indices = np.concatenate([np.arange(down_start, down_end), np.arange(up_start, up_end)])
csi_selected = csi_array[:, :, indices, :]
'''

start_idx = (1024 - 96) // 2  # 464
end_idx = start_idx + 96      # 560
csi_middle_96 = csi_array[:, :, start_idx:end_idx, :]  # [18602, 32, 96, 2]

# 步骤2: 按20个一组分组，不足丢弃
group_size = 20
num_complete_groups = len(csi_middle_96) // group_size
csi_trimmed = csi_middle_96[:num_complete_groups * group_size]  # [x*20, 32, 96, 2]

# 步骤3: 重塑为 [x, 20, 96, 32, 2]
csi_reshaped = csi_trimmed.reshape(num_complete_groups, group_size, 32, 96, 2)
csi_reshaped = np.transpose(csi_reshaped, (0, 1, 3, 2, 4))  # [x, 20, 96, 32, 2]

# 步骤4: 转换为复数矩阵
csi_final = csi_reshaped[..., 0] + 1j * csi_reshaped[..., 1]  # [x, 20, 96, 32]

print(f"最终csi_array形状: {csi_final.shape}")  # [x, 20, 96, 32]

# 保存为.mat文件
mat_data = {
    # 'cfo': cfo_array,
    'csi': csi_array,
    # 'gt_interp_age_tachy': gt_interp_age_tachy_array,
    # 'pos_tachy': pos_tachy_array,
    # 'snr': snr_array,
    # 'time': time_array
}

sio.savemat('dichasus_data.mat', mat_data)
print(f"成功保存数据到 dichasus_data.mat")
print(f"数据形状:")
# print(f"CFO: {cfo_array.shape}")
print(f"CSI: {csi_array.shape}")
# print(f"Position: {pos_tachy_array.shape}")

pass
