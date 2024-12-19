%%%% DeepMIMO parameters set %%%%
% Ray-tracing scenario
params.scenario = 'I1_2p5';        % 上行数据集 (Band1: 2.4 GHz)
params.scene_first = 1;
params.scene_last = 1;

% Active base stations
params.active_BS = [1];            % 选取基站1

% Active users
params.active_user_first = 1;      % 第一行用户
params.active_user_last = 500;     % 根据需要覆盖 100,000 用户
params.enable_BS2BSchannels = 0;
% Subsampling of active users
params.row_subsampling = 1;        % 激活所有用户
params.user_subsampling = 1;

% 天线配置 (SISO)
params.num_ant_BS = [1, 1, 1];     % 基站：单天线
params.num_ant_UE = [1, 1, 1];     % 用户：单天线

% 天线间隔
params.ant_spacing_BS = 0.5;       % 半波长天线间隔
params.ant_spacing_UE = 0.5;

% System parameters
params.bandwidth = 0.5;            % 带宽 0.5 GHz
params.num_paths = 5;              % 路径数：5
params.activate_RX_filter = 0;     % 不启用接收滤波器

% OFDM parameters
params.generate_OFDM_channels = 1; % 启用 OFDM 信道生成
params.num_OFDM = 64;              % 子载波数：64
params.OFDM_sampling_factor = 1;
params.OFDM_limit = 64;

% 保存数据集
params.saveDataset = 1;            % 保存数据集

% % 生成上行
% params.scenario = 'I1_2p4';  % 2.4 GHz (Band1)
% dataset_up = DeepMIMO_Dataset_Generator(params);
% save('uplink_SISO_dataset.mat', 'dataset_up');
% 
% % 生成下行
% params.scenario = 'I1_2p5';  % 2.5 GHz (Band2)
% dataset_down = DeepMIMO_Dataset_Generator(params);
% save('downlink_SISO_dataset.mat', 'dataset_down');


