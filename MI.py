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


def calculate_mi_metrics_multidim(x_raw, y_raw, x_feat=48, y_feat=25):
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
        mi_single = mutual_info_regression(X_s, Y_s[:, i], random_state=42)[0]
        mi_list.append(mi_single)

    mi_avg = np.mean(mi_list)

    # 3. 计算自信息（归一化基准）
    # 计算 I(X; X) 的代理值
    h_x = np.mean([mutual_info_regression(X_s, X_s[:, j], random_state=42)[0]
                   for j in np.random.choice(X_s.shape[1], 5)])

    # 计算 I(Y; Y) 的代理值
    h_y = np.mean([mutual_info_regression(Y_s, Y_s[:, j], random_state=42)[0]
                   for j in np.random.choice(Y_s.shape[1], 5)])

    # 4. 得到归一化互信息 NMI
    ret_nmi = mi_avg / (np.sqrt(h_x * h_y) + 1e-9)
    print(f"{mi_avg:.3g}, {h_x:.3g}, {h_y:.3g}")

    return ret_nmi

    return np.clip(ret_nmi, 0, 1)


# --- 示例：如何针对你的数据调用 ---
# 假设 csi 形状 (ue, speed, 4, 48, 4, 4, 2)
# 假设 aoa 形状 (ue, speed, 25)
# nmi = calculate_mi_metrics_multidim(csi, aoa, x_feat=48, y_feat=25)

def calculate_mi_metrics(x, y):
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


# --- 1. 数据加载 ---
aoa = hdf5storage.loadmat("dataset_generation/AoA.mat")["AoA"]
raw_prev_freq = hdf5storage.loadmat("dataset_generation/H_U_his.mat")["H_U_his"]
raw_pred_freq = hdf5storage.loadmat("dataset_generation/H_D_pre.mat")["H_D_pre"]

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
f_UL = raw_prev_freq[UE_idx, v_idx, -4:]   # (4, 48, 4, 4, 2)
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

print("\n" + "=" * 65)
print(f"{'Domain / Metric':<30} | {'MI (nats)':<10}")
print("-" * 65)


# 遍历
# mi_matrix = np.zeros((f_UL.shape[0], f_UL.shape[1]))
# for speed in range(f_UL.shape[1]):
#     for ue in range(f_UL.shape[0]):
#         mi = calculate_mi_metrics(f_UL[ue, speed], tau_UL[ue, speed])
#         mi_matrix[ue, speed] = mi

print("-" * 65)

# for ue in range(f_UL.shape[0]):
#     mi = calculate_mi_metrics(f_UL[ue, 4], f_DL[ue, 4])
#     print(f"{'Freq: UL vs Freq: DL(UE {})'.format(ue):<30} | {mi:<10.4f}")
# print("-" * 65)

print("test")
# mi = calculate_mi_metrics(f_UL, f_UL)
# print(f"{'test: UL':<30} | {mi:<10.4f}")
#
# mi = calculate_mi_metrics(tau_UL, tau_UL)
# print(f"{'test: UL':<30} | {mi:<10.4f}")

print("-" * 65)
print("跨链路")
mi = calculate_mi_metrics(f_UL, f_DL)
print(f"{'Freq: UL vs Freq: DL':<30} | {mi:<10.4f}")

mi = calculate_mi_metrics(tau_UL, tau_DL)
print(f"{'time: UL vs time: DL':<30} | {mi:<10.4f}")

print("-" * 65)
print("跨模态")
# mi = calculate_mi_metrics(f_UL, tau_UL)
# print(f"{'Freq: UL vs time: UL':<30} | {mi:<10.4f}")

nmi = calculate_mi_metrics_multidim(f_UL, aoa, x_feat=48, y_feat=25)
print(f"{'NMI Freq: UL vs AoA: UL)':<30} | {nmi:<10.4f}")

nmi = calculate_mi_metrics_multidim(f_UL, tau_UL, x_feat=48, y_feat=48)
print(f"{'NMI Freq: UL vs time: UL':<30} | {nmi:<10.4f}")

nmi = calculate_mi_metrics_multidim(tau_UL, aoa, x_feat=48, y_feat=25)
print(f"{'NMI time: UL vs AoA: UL':<30} | {nmi:<10.4f}")

print("-" * 65)
print("freq内部")
# --- 4. 空间维度 (Spatial) ---
for ant in range(4):
    mi = calculate_mi_metrics(f_UL[:, :, 0, 0], f_UL[:, :, 0, ant])
    print(f"{f'Spatial (Antenna {ant})':<30} | {mi:<10.4f}")

# s1 = f_UL[:, :, 0, 0]
# s2 = f_UL[:, :, 0, 1]
# mi_s = calculate_mi_metrics(s1, s2)
# print(f"{'Spatial (Antenna)':<30} | {mi_s:<10.4f}")
#
# # --- 5. 频率维度 (Frequency) ---
# f1 = f_UL[:, :, 0, 0, 0]
# f2 = f_UL[:, :, 1, 0, 0]  # 相邻子载波
# mi_f = calculate_mi_metrics(f1, f2)
# print(f"{'Frequency (Subc)':<30} | {mi_f:<10.4f}")
#
# # --- 6. 时延域分析 (time Domain - 核心新增) ---
# # 比较第 1 个时延分量（通常是强径）与第 5 个时延分量（多径）
# d1 = tau_UL[:, :, 0, 0, 0]
# d2 = tau_UL[:, :, 4, 0, 0]
# mi_d = calculate_mi_metrics(d1, d2)
# print(f"{'time (Tap 1 vs 5)':<30} | {mi_d:<10.4f}")
#
# # --- 7. 时间维度 (Temporal) ---
# t1 = f_UL[-4, :, 0, 0]
# t2 = f_UL[-1, :, 0, 0]
# mi_t = calculate_mi_metrics(t1, t2)
# print(f"{'Temporal (Time)':<30} | {mi_t:<10.4f}")

print("=" * 65)

