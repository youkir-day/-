% ========================================================
% 主程序：Lasso稀疏化RBF源项 + 二维泊松方程求解
% ========================================================
clear; clc; close all;

%% 第一部分：公共参数定义
n_poisson = 20;          % 泊松方程内部节点数 (20x20网格)
h_poisson = 1/(n_poisson+1); % 泊松方程步长

% 源项参数
A1 = 10; A2 = 10;       % 高斯幅值
sigma_sq = 0.02;        % 方差参数
center1 = [0.25, 0.25]; % 高斯中心1
center2 = [0.75, 0.75]; % 高斯中心2

% RBF稀疏化参数
Nx_rbf = 30; Ny_rbf = 30; % 初始中心点网格

%% 第二部分：生成泊松方程的右端项（完整双高斯源）
% 生成泊松方程评估网格
[X_poisson, Y_poisson] = meshgrid(h_poisson:h_poisson:1-h_poisson);
X_test_poisson = [X_poisson(:), Y_poisson(:)]; % 转为Nx2矩阵

% 计算完整右端项f_full
F = @(X) function_F2(X); % 调用目标函数
f_full = F(X_test_poisson);

%% 第三部分：形状参数优化
% 生成初始中心点网格
[xk_x, xk_y] = meshgrid(linspace(0,1,Nx_rbf), linspace(0,1,Ny_rbf));
X_centers_rbf = [xk_x(:), xk_y(:)]; % (Nx_rbf*Ny_rbf)x2

% 计算二维距离矩阵
DM =  pdist2(X_test_poisson, X_centers_rbf, 'euclidean');

% 定义RBF函数（高斯核）
rbf = @(ep, r) exp(-ep*(r).^2);

% LOOCV参数设置
epsilon_list = linspace(0.5, 40, 30); % 候选epsilon范围
cv_errors = zeros(size(epsilon_list));

% LOOCV主循环
for i = 1:length(epsilon_list)
    epsilon = epsilon_list(i);
    Phi = rbf(epsilon, DM);
    
    % 计算LOOCV误差
    H = Phi * pinv(Phi); % 帽子矩阵
    H_ii = diag(H);
    residual_loo = (f_full - Phi * (pinv(Phi) * f_full)) ./ (1 - H_ii);
    cv_errors(i) = mean(residual_loo.^2);
end

% 选择最优epsilon
[~, idx] = min(cv_errors);
epsilon_opt = epsilon_list(idx);
fprintf('LOOCV优化结果：最优epsilon=%.4f\n', epsilon_opt);




%% 第四部分：Lasso稀疏化右端项

% 计算距离矩阵
dist_matrix = pdist2(X_test_poisson, X_centers_rbf, 'euclidean');
A = exp(-epsilon_opt * (dist_matrix).^2);

% Lasso回归（5折交叉验证）
[W, FitInfo] = lasso(A, f_full, 'CV', 5);

% 选择最优lambda
lambda_opt = FitInfo.Lambda1SE;
fprintf('最优lambda: %.6f\n', lambda_opt);

% 获取稀疏权重
w = W(:, FitInfo.Index1SE);
select_idx = find(abs(w) > 1e-6); % 选择有效系数
X_centers_selected = X_centers_rbf(select_idx, :);

% 构造稀疏基矩阵
A_sparse = A(:, select_idx);
% QR分解
[Q,R]=qr(A_sparse,0);
% 最小二乘修正权重
w_sparse =R \ (Q'*f_full);

% 生成稀疏源项
f_sparse = A_sparse * w_sparse;


%% 第五部分：求解泊松方程
% 生成刚度矩阵
S = DiscretePoisson2D(n_poisson);

% 使用稀疏源项求解
b = f_sparse; % 右端项
L = chol(S, 'lower'); % Cholesky分解
v = L \ b;       % 前向替换
w_solve = L' \ v; % 后向替换
u = h_poisson^2 * w_solve; % 缩放解

u_full = h_poisson^2 * (S \ f_full); % 完整解

% 转换解场格式
Z = zeros(n_poisson+2);
Z(2:end-1, 2:end-1) = reshape(u, n_poisson, n_poisson)';
Z_full = zeros(n_poisson+2);
Z_full(2:end-1, 2:end-1) = reshape(u_full, n_poisson, n_poisson)';

%% 第六部分：可视化与误差分析
% 1. LOOCV误差曲线
figure;
semilogy(epsilon_list, cv_errors, 'bo-', 'LineWidth', 1.5);
xlabel('形状参数ε'); ylabel('LOOCV误差');
title('LOOCV误差随ε变化曲线');
grid on;
% 1. 源项对比
figure;
surf(X_poisson, Y_poisson, reshape(f_full, n_poisson, n_poisson));
title('完整源项 f_full'); colorbar;

figure;
surf(X_poisson, Y_poisson, reshape(f_sparse, n_poisson, n_poisson));
title('Lasso稀疏源项 f_sparse'); colorbar;

% 2. 解场对比
x_plot = 0:h_poisson:1;
y_plot = 0:h_poisson:1;
figure;
surf(x_plot, y_plot, Z);
title(' u_{full},u(x1,x2),N=20'); 
colorbar;
view(3);

figure;
surf(x_plot, y_plot, Z_full);
title(' u_{sparse},u(x1,x2),N=20'); 
colorbar;
view(3);

% 3. 误差分析
abs_error_f = abs(f_full - f_sparse);
abs_error_u = abs(reshape(u_full, [], 1) - u);

fprintf('源项最大绝对误差: %.2e\n', max(abs_error_f));
fprintf('源项均方根误差: %.2e\n', sqrt(mean(abs_error_f.^2)));
fprintf('解场最大绝对误差: %.2e\n', max(abs_error_u));
fprintf('解场均方根误差: %.2e\n', sqrt(mean(abs_error_u.^2)));
fprintf('使用基函数数量: %d\n', length(select_idx));

