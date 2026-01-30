import numpy as np
import hdf5storage
from sklearn.feature_selection import mutual_info_regression
import pickle


def preprocess_to_2d(data, feature_dim):
    """
    将高维张量转换为 (N, D) 矩阵。
    N 是所有非特征维度的乘积（样本数），D 是特征维度（48或25）。
    """
    # 找到特征维度所在的位置
    shape = list(data.shape)
    if feature_dim not in shape:
        raise ValueError(f"数据中找不到维度 {feature_dim}")

    f_idx = shape.index(feature_dim)

    # 将特征维度移动到最后一维
    data_moved = np.moveaxis(data, f_idx, -1)

    # 转换为实数（如果是复数或带有实虚部维度）
    data_abs = np.abs(data_moved)

    # 展平所有非特征维度作为样本维度 N
    # 结果形状：(N, feature_dim)
    res = data_abs.reshape(-1, feature_dim)
    return res


def calculate_nmi(x_raw, y_raw, x_feat=48, y_feat=25):
    """
    专门处理多维异构数据的 NMI 计算
    x_raw: (..., 48, ...)
    y_raw: (..., 25, ...)
    """
    # 1. 转换形状为 (N, D)
    # 确保 X 和 Y 的样本总数 N 是一致的
    X = preprocess_to_2d(x_raw, x_feat)
    Y = preprocess_to_2d(y_raw, y_feat)

    # 确保样本对齐（取最小样本数）
    n_samples = min(X.shape[0], Y.shape[0])
    X, Y = X[:n_samples], Y[:n_samples]

    # 随机采样以提速
    sample_size = min(n_samples, 8000)
    idx = np.random.choice(n_samples, sample_size, replace=False)
    X_s, Y_s = X[idx], Y[idx]

    # 2. 计算 I(X; Y) - 遍历 Y 的特征维度
    mi_list = []
    # 为了提速，如果特征维度太多（如48），可以间隔采样或只算部分维度
    for i in range(Y_s.shape[1]):
        # X 是多维特征 (N, 48), Y_s[:, i] 是单维目标 (N, 1)
        mi_single = mutual_info_regression(X_s, Y_s[:, i].T, random_state=42)[0]
        mi_list.append(mi_single)

    mi_avg = np.mean(mi_list)

    # 3. 计算自信息（归一化基准）
    # 计算 I(X; X) 的代理值
    h_x = np.mean([mutual_info_regression(X_s, X_s[:, j], random_state=42)[0]
                   for j in np.random.choice(X_s.shape[1], 15)])

    # 计算 I(Y; Y) 的代理值
    h_y = np.mean([mutual_info_regression(Y_s, Y_s[:, j], random_state=42)[0]
                   for j in np.random.choice(Y_s.shape[1], 15)])

    # 4. 得到归一化互信息 NMI
    ret_nmi = mi_avg / (np.sqrt(h_x * h_y) + 1e-9)
    print(f"{mi_avg:.3g}, {h_x:.3g}, {h_y:.3g}")

    return ret_nmi, mi_avg


# --- 示例：如何针对你的数据调用 ---
# 假设 csi 形状 (ue, speed, 4, 48, 4, 4, 2)
# 假设 aoa 形状 (ue, speed, 25)
# nmi = calculate_mi_metrics_multidim(csi, aoa, x_feat=48, y_feat=25)

def calculate_mi(x, y):
    """
    计算互信息 (MI)
    """
    x_val = np.abs(x).flatten().reshape(-1, 1)
    y_val = np.abs(y).flatten()

    sample_size = min(len(x_val), 10000)
    idx = np.random.choice(len(x_val), sample_size, replace=False)
    x_val, y_val = x_val[idx], y_val[idx]

    mi = mutual_info_regression(x_val, y_val, random_state=42)[0]
    # # h_x = 0.5 * np.log(2 * np.pi * np.e * (np.var(x_val) + 1e-9))
    # h_y = 0.5 * np.log(2 * np.pi * np.e * (np.var(y_val) + 1e-9))

    return mi


def preprocess_new_data(data_96d):
    """
    将96维（实部和虚部拼接）数据转换为48维复数格式
    data_96d: shape (..., 96)
    返回: shape (..., 48) 复数数组
    """
    shape = list(data_96d.shape)
    shape[-1] = 48  # 最后一维从96变为48

    # 拆分实部和虚部
    real_part = data_96d[..., :48]
    imag_part = data_96d[..., 48:]

    # 组合成复数
    data_complex = real_part + 1j * imag_part
    return data_complex


# --- 1. 数据加载 ---
aoa = hdf5storage.loadmat("dataset_generation/AoA.mat")["AoA"]
raw_prev_freq = hdf5storage.loadmat("dataset_generation/H_U_his.mat")["H_U_his"]
raw_pred_freq = hdf5storage.loadmat("dataset_generation/H_D_pre.mat")["H_D_pre"]

# 加载新数据（来自10_25dB.mat）
new_data = hdf5storage.loadmat("dataset_generation/10_20dB.mat")
ground_truth_96d = new_data["ground_truth"]  # (31, 1024, 4, 96)
model_output_96d = new_data["model_output"]  # (31, 1024, 4, 96)
prev_96d = new_data["prev"][:, :, -4:, :]

print("ground_truth_96d shape:", ground_truth_96d.shape)
print("model_output_96d shape:", model_output_96d.shape)

# 将96维数据转换为48维复数格式
ground_truth_48d = preprocess_new_data(ground_truth_96d)  # (31, 1024, 4, 48) complex
model_output_48d = preprocess_new_data(model_output_96d)  # (31, 1024, 4, 48) complex
prev_48d = preprocess_new_data(prev_96d)

# (注意：后续会根据f_UL/f_DL的实际大小调整新数据的维度)

# (保留你原始的 pickle 加载逻辑)
# with open("Testing Dataset/H_U_his_test.pickle", "rb") as f:
#     raw_prev_freq = pickle.load(f)
#     print('raw_prev_freq shape', raw_prev_freq.shape)
# with open("Testing Dataset/H_D_pre_test.pickle", "rb") as f:
#     raw_pred_freq = pickle.load(f)
#     print('raw_pred_freq shape', raw_pred_freq.shape)

# 选定一个中间速度场景进行分析
v_idx = 0  # slice(None) # magic_number:98
UE_idx = 0  # slice(None)

# 速度，历史信道探测次数，子载波数，天线数（垂直），天线数（水平），极化方向
f_UL = raw_prev_freq[UE_idx, v_idx, -4:]  # (4, 48, 4, 4, 2)
f_DL = raw_pred_freq[UE_idx, v_idx]
print("f_UL shape", f_UL.shape)
print("f_DL shape", f_DL.shape)

# 执行 IDFT 变换到时延域 (Delay Domain)
# (1000, 4, 48, 4, 4, 2)
# 速度，历史信道探测次数，采样点数，天线数（垂直），天线数（水平），极化方向
tau_UL = np.fft.ifft(f_UL, axis=1)  # (4, 48, 4, 4, 2)
tau_DL = np.fft.ifft(f_DL, axis=1)
print("tau_UL shape", tau_UL.shape)
print("tau_DL shape", tau_DL.shape)

# --- 调整新数据维度，使其元素总数与f_UL/f_DL一致 ---
target_size = f_UL.size  # 6144
print(f"\nf_UL size: {f_UL.size}, f_DL size: {f_DL.size}")

# f_UL/f_DL的shape: (4, 48, 4, 4, 2), 除了48维外其他维度的乘积 = 4 * 4 * 4 * 2 = 128
# 新数据需要调整为 (..., 48)，其中其他维度乘积 = 128

feat_dim = 48

# 先将ground_truth_48d展平为 (total_samples, 48)
shape_gt = list(ground_truth_48d.shape)
feat_idx_gt = shape_gt.index(feat_dim)
ground_truth_moved = np.moveaxis(ground_truth_48d, feat_idx_gt, -1)
ground_truth_flat = ground_truth_moved.reshape(-1, feat_dim)

# 同样处理model_output_48d
shape_mo = list(model_output_48d.shape)
feat_idx_mo = shape_mo.index(feat_dim)
model_output_moved = np.moveaxis(model_output_48d, feat_idx_mo, -1)
model_output_flat = model_output_moved.reshape(-1, feat_dim)

shape_prev = list(prev_48d.shape)
feat_idx_prev = shape_prev.index(feat_dim)
prev_moved = np.moveaxis(prev_48d, feat_idx_prev, -1)
prev_flat = prev_moved.reshape(-1, feat_dim)

print(f"ground_truth_flat shape: {ground_truth_flat.shape}")
print(f"model_output_flat shape: {model_output_flat.shape}")
print(f"prev_flat shape:{prev_flat.shape}")

# 计算需要的样本数
target_samples = target_size // feat_dim  # 128

# 生成一次随机索引，确保两个数据采样一致
indices = np.random.choice(ground_truth_flat.shape[0], target_samples, replace=False)

# 使用相同索引采样
ground_truth_sampled = ground_truth_flat[indices]
model_output_sampled = model_output_flat[indices]
prev_sampled = prev_flat[indices]
prev_tau_flat = np.fft.ifft(prev_flat, axis=-1)
prev_tau_sampled = prev_tau_flat[indices]

# reshape为 (4, 4, 8, 48) 以匹配类似f_UL/f_DL的结构
# 4 * 4 * 8 = 128，符合要求
ground_truth_48d = ground_truth_sampled.reshape(4, 4, 8, feat_dim)
model_output_48d = model_output_sampled.reshape(4, 4, 8, feat_dim)
prev_output_48d = prev_sampled.reshape(4, 4, 8, feat_dim)
prev_tau_output_48d = prev_tau_sampled.reshape(4, 4, 8, feat_dim)


print(f"ground_truth_48d shape: {ground_truth_48d.shape}, size: {ground_truth_48d.size}")
print(f"model_output_48d shape: {model_output_48d.shape}, size: {model_output_48d.size}")
print(f"prev_output_48d shape: {prev_output_48d.shape}, size: {prev_output_48d.size}")
print(f"prev_tau_output_48d shape: {prev_tau_output_48d.shape}, size: {prev_tau_output_48d.size}")

print("\n" + "=" * 65)
print(f"{'Domain / Metric':<30} | {'MI (nats)':<10}")
print("-" * 65)

# print("跨模态")
#
# nmi, _ = calculate_nmi(f_UL, aoa, x_feat=48, y_feat=25)
# print(f"{'NMI Freq: UL vs AoA: UL':<30} | {nmi:<10.4f}")
#
# nmi, _ = calculate_nmi(f_UL, tau_UL, x_feat=48, y_feat=48)
# print(f"{'NMI Freq: UL vs time: UL':<30} | {nmi:<10.4f}")
#
# nmi, _ = calculate_nmi(tau_UL, aoa, x_feat=48, y_feat=25)
# print(f"{'NMI time: UL vs AoA: UL':<30} | {nmi:<10.4f}")

# print("-" * 65)

mi_hat_gt = np.zeros(128)
mi_prev_gt = np.zeros(128)
mi_tau_gt = np.zeros(128)
for index in range(128):
    mi_hat_gt[index] = calculate_mi(ground_truth_flat[index, :], model_output_flat[index, :])
    mi_prev_gt[index] = calculate_mi(prev_flat.reshape(-1, 48)[index], model_output_flat.reshape(-1, 48)[index, :])
    mi_tau_gt[index] = calculate_mi(prev_tau_flat.reshape(-1, 48)[index], ground_truth_flat.reshape(-1, 48)[index, :])
print(f"{np.mean(mi_hat_gt):<4f}")
print(f"{np.mean(mi_prev_gt):<4f}")
print(f"{np.mean(mi_tau_gt):<4f}")

#
# print("跨链路")
_, mi = calculate_nmi(f_UL, ground_truth_48d, x_feat=48, y_feat=48)
print(f"{'Freq: UL vs Freq: DL':<30} | {mi:<10.4f}")

_, mi = calculate_nmi(tau_UL, ground_truth_48d, x_feat=48, y_feat=48)
print(f"{'time: UL vs Freq: DL':<30} | {mi:<10.4f}")

_, mi = calculate_nmi(aoa, ground_truth_48d, x_feat=25, y_feat=48)
print(f"{'AoA:  UL vs Freq: DL':<30} | {mi:<10.4f}")


print("-" * 65)

# --- 新数据的互信息计算 ---

# 2.1 新数据自身之间的互信息
print("新数据 (10_25dB.mat):")

# Ground truth vs Model output（预测质量评估）
_, mi = calculate_nmi(model_output_48d, ground_truth_48d, x_feat=48, y_feat=48)
print(f"{'Ground truth vs Model output':<30} | {mi:<10.4f}")
