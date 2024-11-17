clear;
folder = "./csi_output/20241116_17_42/";

for speed = 10:10:100
    filename = sprintf(folder + "%d.mat",speed);
    load(filename);
    sizeA = size(ground_truth); % 获取ground_truth的尺寸，假设为 [a, b, c, d]

    a = sizeA(1); 
    b = sizeA(2);
    c = sizeA(3);
    d = sizeA(4);
    
    % 重新调整维度
    ground_truth = reshape(ground_truth, [a*b, c, d]);
    model_output = reshape(model_output, [a*b, c, d]);
    
    % 将实部和虚部组合为复数矩阵
    complex_ground_truth = ground_truth(:, :, 1:48) + ...
        1i * ground_truth(:, :, 49:96); % 尺寸为 [a*b, c, 48]
    complex_model_output = model_output(:, :, 1:48) + ...
        1i * model_output(:, :, 49:96); % 尺寸为 [a*b, c, 48]
    
    savename = sprintf(folder + "complex_%d.mat",speed);
    save(savename,'complex_ground_truth','complex_model_output');

end
