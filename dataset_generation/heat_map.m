% --- MATLAB Code: Read CSV and Plot Heatmap ---
close all;
% 1. 读取CSV文件
%    假设你的CSV文件名为 'your_data_file.csv'
%    将 'your_data_file.csv' 替换为你实际的文件名。
filename = 'mi_matrix_ful_tauul.csv';

% 使用 readmatrix 读取纯数值数据 (推荐)
% 如果你的CSV包含标题行，readmatrix 默认会跳过它们（如果第一行是标题且与数据类型不同）
try
    data_matrix = readmatrix(filename);
    fprintf('成功读取文件: %s\n', filename);
catch ME
    % 如果 readmatrix 失败，可能是由于混合数据类型（如标题行），尝试其他方法
    fprintf('使用 readmatrix 读取失败: %s\n', ME.message);
    
    % 如果文件有标题行，可以使用 readtable
    % 这里假设第一行是标题，第一列也是标签（例如行名）
    try
        data_table = readtable(filename);
        fprintf('改用 readtable 成功读取: %s\n', filename);
        
        % 提取数值部分，通常是从第二列开始到最后一列
        % 如果第一列不是标签而是数据，请移除 data_table{:, 2:end} 中的 2:
        data_matrix = table2array(data_table{:, 2:end});
        
        % 可选：提取行名和列名用于热力图标签
        row_names = data_table.Properties.VariableNames(2:end); % 列名作为 Y 轴标签
        col_names = data_table{:, 1}; % 第一列作为 X 轴标签
        
    catch ME2
        error('无法读取文件: %s。错误: %s', filename, ME2.message);
    end
end

% 检查数据是否成功加载
if isempty(data_matrix)
    error('读取的数据为空。请检查文件 "%s" 是否存在且包含有效数据。', filename);
end

fprintf('数据维度: %d 行 x %d 列\n', size(data_matrix));

% 2. 绘制热力图
figure; % 创建一个新的图形窗口

% 方案 A: 如果使用 readtable 读取，并且有行列标签
if exist('row_names', 'var') && exist('col_names', 'var')
    h = heatmap(col_names, row_names, data_matrix); % X轴是列名，Y轴是行名
else
    % 方案 B: 如果只有数值矩阵 (来自 readmatrix 或 table2array)
    % MATLAB 会自动使用数字索引作为轴标签
    h = heatmap(data_matrix(:,2:end));
end

% 3. 自定义热力图外观
title(sprintf('Heatmap of %s', filename));
xlabel('speed'); % 如果使用方案B
ylabel('ue');   % 如果使用方案B
% 如果使用方案A，xlabel 和 ylabel 会由 heatmap 函数根据输入的标签自动设置

% 可选：更改颜色映射
% h.Colormap = parula; % 更改为 parula 颜色
% h.Colormap = hot;    % 更改为 hot 颜色

% 可选：反转 Y 轴方向 (让第一行显示在顶部)
% h.YDisplayData = fliplr(h.YDisplayData); % 这样做可能不直接生效
% 更常用的方法是在绘制前翻转矩阵
% data_matrix = flipud(data_matrix); % 翻转矩阵
% 然后重新绘制: heatmap(...)

% 4. 显示图形
% figure 窗口已经打开，图形已绘制，无需额外命令即可看到结果。

% --- End of Script ---

