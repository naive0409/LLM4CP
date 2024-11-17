% 定义数据
speed = 10:10:100; % 速度 (x轴)

% 模拟的 NMSE 数据 (y轴)
% 241115_22_00
% clip = [0.024045301120608084,0.02576213131748861,0.027639086748803814,0.028201536365574407,0.0328985971188353,0.036011886572645556,0.04377718984840378,0.04798303460401873,0.058473670674908544,0.06758702906870073];
% 20241116_17_42
clip = [0.518686935786278	0.5023490286642506	0.5002536725613379	0.5107519097866551	0.483438850410523	0.49971654049811826	0.5218169602655596	0.47490900466519015	0.5006027135156816	0.5142723033505101];
% 绘制对数坐标图
figure;

semilogy(speed, clip, '-v','Color', [0.75 0 0.75], 'LineWidth', 1.5);

% 图例与标签
legend({'clip'}, ...
        'Location', 'northeast', 'FontSize', 12);
       % 'Location', 'southeast', 'FontSize', 12);
xlabel('Speed (km/h)', 'FontSize', 14);
ylabel('NMSE', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% 锁定横纵坐标范围
xlim([10 100]); % 横轴范围
% ylim([1e-2 2]); % 纵轴范围llm4cp tdd
ylim([0.35 2]); % 纵轴范围llm4cp fdd

set(gca, 'Color', 'none');  % 设置当前坐标轴背景透明
% exportgraphics(gcf, 'tdd_nmse.png', 'BackgroundColor', 'none');
exportgraphics(gcf, 'fdd_nmse.png', 'BackgroundColor', 'none');