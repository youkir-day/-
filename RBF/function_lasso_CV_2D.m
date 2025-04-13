clear;
clc;

% 参数设置
Nx = 30; 
Ny = 30;
n = 50;

% 生成二维中心点
[xk_x, xk_y] = meshgrid(linspace(0,1,Nx), linspace(0,1,Ny));
X_center = [xk_x(:), xk_y(:)];  % 转换为(Nx*Ny)x2的中心坐标

% 生成二维测试点
[xe_x, xe_y] = meshgrid(linspace(0,1,n), linspace(0,1,n));
X_test = [xe_x(:), xe_y(:)];   

% 二维目标函数
F = @(X)function_F2(X);
y = F(X_test);

% RBF参数设置
epsilon = 10;

% 计算距离矩阵
dist_matrix = pdist2(X_test, X_center, 'euclidean');%欧几里得距离
A = exp(-epsilon * (dist_matrix).^2);

% 使用lasso进行交叉验证，5折交叉验证
[W, FitInfo] = lasso(A, y, 'CV', 5);

% 选择最优lambda
lambda_option = FitInfo.Lambda1SE;
fprintf('最优lambda为：%.8f\n', lambda_option);

% 使用最优lambda重新训练
w = W(:, FitInfo.Index1SE);
select_idx = find(w >1e-6);
select_xk = X_center(select_idx, :);

% 构造稀疏模型
A_sparse = A(:, select_idx);
w_sparse = A_sparse \ y;       % 最小二乘求解
fe = A_sparse * w_sparse;

% 误差分析
mse = mean((y - fe).^2);
error = max(abs(fe - y));
fprintf('均方误差mse: %.2e\n', mse);
fprintf('最大绝对误差: %.2e\n', error);
fprintf('使用基函数数量: %d\n', length(select_idx));

%% 可视化结果
ZZ_true = reshape(y, [n, n]);
ZZ_pred = reshape(fe, [n, n]);

% 绘制真实函数和lasso稀疏化RBF近似曲面
figure;
subplot(1,2,1);
surf(xe_x, xe_y, ZZ_true, 'EdgeColor', 'none'); % 真实函数曲面
title('真实函数');
xlabel('x1'); ylabel('x2'); zlabel('F');

subplot(1,2,2);
surf(xe_x, xe_y, ZZ_pred, 'EdgeColor', 'none'); % lasso稀疏化RBF近似曲面
title('lasso稀疏化RBF近似');
xlabel('x1'); ylabel('x2'); zlabel('F');




