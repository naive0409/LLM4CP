%% down
clear;
load('/Users/wu/dev/LLM4CP/DEEPMIMOI1/DeepMIMO_dataset/dataset_down.mat')

length(DeepMIMO_dataset{1}.user);
combined_channel_down = cell(1,145600); %145600 20x48x...
cell_index = 1;

channels = cellfun(@(u) squeeze(u.channel)', DeepMIMO_dataset{1}.user(:), 'UniformOutput', false);

for row_index = 1:400 % 400行
    row_offset = 1 + 201 * (row_index - 1); % 1, 202, 1+2*202, ...
    for time_index=20:201 % 201列
        idx_1 = row_offset + time_index - 20;
        idx_2 = row_offset + time_index - 1;
        timeseries = cat(1, channels{idx_1:idx_2});
        % 16+4=20个按时间排序的序列
        % combined_channel_down = [combined_channel_down,{timeseries},{flipud(timeseries)}];
        combined_channel_down{cell_index} = timeseries; cell_index = cell_index + 1;
        combined_channel_down{cell_index} = flipud(timeseries); cell_index = cell_index + 1;
        % 正序倒序都算
    end
end

c_down = cat(3, combined_channel_down{:});


%% up 流程完全一样
clear;
load('/Users/wu/dev/LLM4CP/DEEPMIMOI1/DeepMIMO_dataset/dataset_up.mat')

length(DeepMIMO_dataset{1}.user);
combined_channel_up = cell(1,145600); %145600 20x48x...
cell_index = 1;

channels = cellfun(@(u) squeeze(u.channel)', DeepMIMO_dataset{1}.user(:), 'UniformOutput', false);

for row_index = 1:400 % 400行
    row_offset = 1 + 201 * (row_index - 1); % 1, 202, 1+2*202, ...
    for time_index=20:201 % 201列
        idx_1 = row_offset + time_index - 20;
        idx_2 = row_offset + time_index - 1;
        timeseries = cat(1, channels{idx_1:idx_2});
        % 16+4=20个按时间排序的序列
        % combined_channel_up = [combined_channel_up,{timeseries},{flipud(timeseries)}];
        combined_channel_up{cell_index} = timeseries; cell_index = cell_index + 1;
        combined_channel_up{cell_index} = flipud(timeseries); cell_index = cell_index + 1;
        % 正序倒序都算
    end
end

c_up = cat(3, combined_channel_up{:});

%% 20 -> 16 + 4
c_down_his = c_down(1:16,:,:);
c_down_pre = c_down(17:20,:,:);
c_up_his = c_up(1:16,:,:);
c_up_pre = c_up(17:20,:,:);

save('DeepMIMO_dataset/c_down_his.mat','c_down_his');
save('DeepMIMO_dataset/c_down_pre.mat','c_down_pre');
save('DeepMIMO_dataset/c_up_his.mat','c_up_his');
save('DeepMIMO_dataset/c_up_pre.mat','c_up_pre');