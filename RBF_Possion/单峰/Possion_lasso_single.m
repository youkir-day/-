% =================================================================
% 主程序：使用LU分解求解二维变系数泊松方程 + Lasso稀疏RBF源项
% 包含原始解和稀疏RBF解的三维视图对比
% =================================================================

close all; clear; clc;

% ========================= 输入参数设置 ==========================
n = 20;         % 单方向内部节点数
a_amp = 12;     % 系数函数a的幅度参数
f_amp = 1;      % 源项f的幅度参数
x_0 = 0.5;      % 高斯分布中心x坐标
y_0 = 0.5;      % 高斯分布中心y坐标
c_x = 1;        % x方向标准差系数
c_y = 1;        % y方向标准差系数
h = 1/(n+1);    % 网格步长计算（区域[0,1]×[0,1]）

% RBF稀疏化参数
Nx_rbf = 30; Ny_rbf = 30; % 初始中心点网格

% ====================== 生成计算网格 ==========================
[X, Y] = meshgrid(h:h:1-h); % 内部节点
X_test_poisson = [X(:), Y(:)];       % 转为Nx2矩阵

% ====================== 生成完整源项 ==========================
f_full = f_amp*exp(-((X(:)-x_0).^2/(2*c_x^2) + (Y(:)-y_0).^2/(2*c_y^2)));

%%  交叉验证选择最优ε形状参数优化
% 生成初始中心点网格
[xk_x, xk_y] = meshgrid(linspace(0,1,Nx_rbf), linspace(0,1,Ny_rbf));
X_centers_rbf = [xk_x(:), xk_y(:)]; % (Nx_rbf*Ny_rbf)x2

% 计算二维距离矩阵
DM =  pdist2(X_test_poisson, X_centers_rbf, 'euclidean');

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
    residual_loo = (f_full - Phi * (pinv(Phi) * f_full)) ./ (1 - H_ii);
    cv_errors(i) = mean(residual_loo.^2);
end

% 选择最优epsilon
[~, idx] = min(cv_errors);
epsilon_opt = epsilon_list(idx);
fprintf('LOOCV优化结果：最优epsilon=%.4f\n', epsilon_opt);
% 选择最优ε
[~, opt_idx] = min(cv_errors);
epsilon_opt = epsilon_list(opt_idx);
fprintf('最优ε=%.2f\n', epsilon_opt);

%% =========lasso稀疏化RBF=========
% 使用最优ε重新计算
dist_matrix = pdist2(X_test_poisson, X_centers_rbf, 'euclidean');
A = exp(-epsilon_opt * (dist_matrix).^2);
[W, FitInfo] = lasso(A, f_full, 'CV', 5, 'Alpha', 1, 'Standardize', false);
lambda_opt = FitInfo.Lambda1SE;
w = W(:, FitInfo.Index1SE);

% 获取稀疏中心点
select_idx = find(abs(w) > 1e-6);
X_centers_selected = X_centers_rbf(select_idx, :);
w_sparse = w(select_idx);

% 重构稀疏源项
A_sparse = exp(-epsilon_opt * pdist2(X_test_poisson, X_centers_selected, 'euclidean').^2);
f_sparse = A_sparse * w_sparse;

% ====================== 矩阵与向量生成 ==========================
% 生成n²×n²的二维泊松方程刚度矩阵（五点差分格式）
S = DiscretePoisson2D(n);

% LU分解（带部分主元选择）
[L, U, P] = lu(S);

% 构建系数矩阵a(x,y)（高斯型变系数）
C = zeros(n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = 1 + a_amp*exp(-((i*h-x_0)^2/(2*c_x^2) + ...
                  (j*h-y_0)^2/(2*c_y^2)));
    end
end

% 创建对角矩阵D（用于系数处理）
D = zeros(n^2,n^2);
for i = 1:n
    for j = 1:n
        D(j+n*(i-1), j+n*(i-1)) = C(i,j);
    end
end

% ====================== 方程求解过程 ==========================
% 使用完整源项求解（原始解）
b_full = zeros(n^2,1);
for i = 1:n
    for j = 1:n
        idx = n*(i-1)+j;
        b_full(idx) = f_full(idx)/C(i,j);
    end
end
v_full = L\(P*b_full);
w_full = U\v_full;
u_full = h^2 * w_full;

% 使用稀疏源项求解
b_sparse = zeros(n^2,1);
for i = 1:n
    for j = 1:n
        idx = n*(i-1)+j;
        b_sparse(idx) = f_sparse(idx)/C(i,j);
    end
end
v_sparse = L\(P*b_sparse);
w_sparse = U\v_sparse;
u_sparse = h^2 * w_sparse;

% ====================== 结果可视化 ==========================
% 生成网格坐标
x1 = 0:h:1;
y1 = 0:h:1;

% 将解向量转换为网格格式（含边界零点）
Z_full = zeros(n+2,n+2);
Z_sparse = zeros(n+2,n+2);
for i = 1:n
    for j = 1:n
        idx = j+n*(i-1);
        Z_full(i+1,j+1) = u_full(idx);
        Z_sparse(i+1,j+1) = u_sparse(idx);
    end
end

% 1. LOOCV误差曲线
figure;
semilogy(epsilon_list, cv_errors, 'bo-', 'LineWidth', 1.5);
xlabel('形状参数ε'); ylabel('LOOCV误差');
title('LOOCV误差随ε变化曲线');
grid on;

% 1.原始解场三维视图
figure;
surf(x1, y1, Z_full);
colorbar;
xlabel('x'), ylabel('y'), zlabel('u(x,y)');
title('u_{full}(x1,x2),A=12,n=20');
% view(2);

% 2.稀疏RBF解场三维视图
figure;
surf(x1, y1, Z_sparse);
colorbar;
xlabel('x'), ylabel('y'), zlabel('u(x,y)');
title('u_{sparse}(x1,x2),A=12,n=20');
% view(2);


% 3.源项对比图
figure;
surf(X, Y, reshape(f_full, n, n));
title('原始源项 f(x,y)');
xlabel('x'), ylabel('y'), zlabel('f(x,y)');

figure;
surf(X, Y, reshape(f_sparse, n, n));
title('稀疏RBF近似 ');
xlabel('x'), ylabel('y'), zlabel('f_{sparse}(x,y)');
%4.误差分析
abs_error = abs(Z_full - Z_sparse);
fprintf('最大绝对误差: %.4e\n', max(abs_error(:)));
fprintf('均方根误差: %.4e\n', sqrt(mean(abs_error(:).^2)));
fprintf('使用RBF中心点数量: %d\n', length(select_idx));

