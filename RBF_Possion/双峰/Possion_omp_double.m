% ========================================================
% 主程序：LOOCV优化形状参数 + 贪心RBF稀疏化 + 解场对比
% ========================================================
clear; clc; close all;

%% 第一部分：公共参数定义
n_poisson = 20;          % 泊松方程内部节点数 (20x20网格)
h_poisson = 1/(n_poisson+1); % 泊松方程步长

% RBF稀疏化参数
Nx_rbf = 30; Ny_rbf = 30; % 初始中心点网格

%% 第二部分：生成泊松方程的右端项
% 生成泊松方程的评估网格
[X_poisson, Y_poisson] = meshgrid(h_poisson:h_poisson:1-h_poisson);
X_test_poisson = [X_poisson(:), Y_poisson(:)]; % 转换为Nx2矩阵

% 计算完整右端项f_full
F = @(X) function_F2(X); % 调用目标函数
f_full = F(X_test_poisson);

%% 第三部分：LOOCV优化形状参数epsilon
% 生成初始中心点网格
[xk_x, xk_y] = meshgrid(linspace(0, 1, Nx_rbf), linspace(0, 1, Ny_rbf));
X_centers_rbf = [xk_x(:), xk_y(:)]; % 初始中心点 (Nx_rbf*Ny_rbf)x2

% 计算二维距离矩阵
DM = pdist2_2D(X_test_poisson, X_centers_rbf);

% 定义RBF函数（高斯核）
rbf = @(ep, r) exp(-ep*(r).^2);

% LOOCV参数设置
epsilon_list = linspace(0.5, 15, 30); % 候选epsilon范围
cv_errors = zeros(size(epsilon_list));

% LOOCV主循环
for i = 1:length(epsilon_list)
    epsilon = epsilon_list(i);
    Phi = rbf(epsilon, DM);
    
    % 计算LOOCV误差
    H = Phi * pinv(Phi); % 帽子矩阵
    H_ii = diag(H);
    residual_loo = (f_full - Phi * (pinv(Phi) *f_full)) ./ (1 - H_ii);
    
    cv_errors(i) = mean(residual_loo.^2);
end

% 选择最优epsilon
[~, idx] = min(cv_errors);
epsilon_opt = epsilon_list(idx);
fprintf('LOOCV优化结果：最优epsilon=%.4f\n', epsilon_opt);

%% 第四部分：贪心RBF稀疏化右端项
I = [];       % 选取的中心点索引
A_rbf = [];   % RBF基函数矩阵
r = f_full;   % 初始残差
prev_cv_error = inf; % LOOCV误差历史

% 贪心算法主循环
for k = 1:Nx_rbf*Ny_rbf
    available = setdiff(1:size(X_centers_rbf,1), I);
    if isempty(available)
        break;
    end
    
    % 1. 选择使残差内积最大的中心点
    max_inner = -inf;
    best_idx = 0;
    for idx = available
        dist = sqrt(sum((X_test_poisson - X_centers_rbf(idx,:)).^2, 2));
        phi = exp(-(epsilon_opt * dist).^2); % 使用优化后的epsilon
        inner = phi' * r;
        if abs(inner) > max_inner
            max_inner = abs(inner);
            best_idx = idx;
        end
    end
    
    % 2. 更新基函数矩阵
    I = [I, best_idx];
    dist_new = sqrt(sum((X_test_poisson - X_centers_rbf(best_idx,:)).^2, 2));
    A_rbf = [A_rbf, exp(-(epsilon_opt * dist_new).^2)];
    
    % 3. QR分解求解权重
    [Q, R] = qr(A_rbf, 0);
    w = R \ (Q' * f_full);
    
    % 4. 更新残差
    r = f_full - A_rbf * w;
    
    % 5. LOOCV误差计算
    H_ii = sum(Q.^2, 2);
    e_loo = r ./ (1 - H_ii);
    cv_error = mean(e_loo.^2);
    
    % 6. 终止条件：LOOCV误差上升
    if cv_error > prev_cv_error
        fprintf('贪心终止：迭代%d，LOOCV误差%.2e\n', k, cv_error);
        break;
    end
    prev_cv_error = cv_error;
end

% 生成稀疏化源项
f_sparse = A_rbf * w;

%% 第五部分：求解泊松方程（完整解和稀疏解对比）
% 生成刚度矩阵
S = DiscretePoisson2D(n_poisson);

% 1. 完整源项的解
b_full=f_full;      %右端项
[L]=chol(S,"lower");    %Cholesky分解
v_full=L\b_full;    %前向替换
w_full=L'\v_full;   %后项替换
u_full=h_poisson^2*w_full;


% 2. 稀疏源项的解
b_sparse=f_sparse;      %右端项
v_sparse=L\b_sparse;    %前向替换
w_sparse=L'\v_sparse;   %后项替换
u_sparse=h_poisson^2*w_sparse;
% 计算解场绝对误差
abs_error_u = abs(reshape(u_full, n_poisson, n_poisson) - reshape(u_sparse, n_poisson, n_poisson));

% 将解转为网格格式（含边界零点）
Z_full = zeros(n_poisson+2);
Z_full(2:end-1, 2:end-1) = reshape(u_full, n_poisson, n_poisson)';

Z_sparse = zeros(n_poisson+2);
Z_sparse(2:end-1, 2:end-1) = reshape(u_sparse, n_poisson, n_poisson)';

%% 第六部分：可视化对比

% 1. LOOCV误差曲线
figure(1);
semilogy(epsilon_list, cv_errors, 'bo-', 'LineWidth', 1.5);
xlabel('形状参数ε'); ylabel('LOOCV误差');
title('LOOCV误差随ε变化曲线');

% 2. 源项对比

figure(2);
surf(X_poisson, Y_poisson, reshape(f_full, n_poisson, n_poisson));
title('完整源项 f_{full}'); 
colorbar; 
view(3);

figure(3);
surf(X_poisson, Y_poisson, reshape(f_sparse, n_poisson, n_poisson));
title(sprintf('稀疏化源项 (ε=%.2f)', epsilon_opt));
colorbar; 
view(3);

% 3. 解场对比
x_plot = 0:h_poisson:1;
y_plot = 0:h_poisson:1;

figure(4);
surf(x_plot, y_plot, Z_full);
title(' u_{full}(x1,x2),N=20'); xlabel('x'); ylabel('y'); 
colorbar; 
view(2);
% view(3);

figure(5);
surf(x_plot, y_plot, Z_sparse);
title(' u_{sparse}(x1,x2),N=20'); xlabel('x'); ylabel('y'); 
colorbar; 
view(2);
% view(3);

%% 第七部分：误差分析
max_error_f = max(abs(f_full - f_sparse));
rmse_f = sqrt(mean((f_full - f_sparse).^2));

u_error = abs(u_full - u_sparse);
max_error_u = max(u_error);
rmse_u = sqrt(mean(u_error.^2));

fprintf('=== 误差分析 ===\n');
fprintf('源项最大绝对误差: %.2e\n', max_error_f);
fprintf('源项均方根误差: %.2e\n', rmse_f);
fprintf('解场最大绝对误差: %.2e\n', max_error_u);
fprintf('解场均方根误差: %.2e\n', rmse_u);
fprintf('使用中心点数: %d\n', numel(I));

%% 辅助函数

function D = pdist2_2D(X, Y)
    % 二维欧氏距离矩阵
    D = sqrt((X(:,1) - Y(:,1)').^2 + (X(:,2) - Y(:,2)').^2);
end
