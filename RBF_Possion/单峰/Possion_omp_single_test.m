% =================================================================
% 主程序：使用LU分解求解二维变系数泊松方程 + 贪心稀疏RBF方法
% 方程形式：-∇·(a(x,y)∇u) = f(x,y)
% 改进说明：
% 1. 使用贪心算法和LOOCV优化RBF近似源项f(x,y)
% 2. 自动选择最优形状参数ε
% 3. 稀疏化处理减少计算量
% =================================================================

close all; clear; clc;

%% ====================== 参数设置 ==========================
n = 20;             % 单方向内部节点数
a_amp = 12;         % 系数函数a的幅度参数
f_amp = 1;          % 源项f的幅度参数
x_0 = 0.5;          % 高斯分布中心x坐标
y_0 = 0.5;          % 高斯分布中心y坐标
c_x = 1;            % x方向标准差系数
c_y = 1;            % y方向标准差系数
h = 1/(n+1);        % 网格步长

% RBF参数
Nx_rbf = 15; Ny_rbf = 15;   % 初始中心点网格
epsilon_list = linspace(0.1, 30, 50); % ε候选范围

%% ====================== 生成系统矩阵 ========================
S = DiscretePoisson2D(n);   % 刚度矩阵
[L, U, P] = lu(S);          % LU分解

%% ====================== 变系数生成 ========================
% 生成计算网格
[X, Y] = meshgrid(h:h:1-h);
X_vec = [X(:), Y(:)];       % 转换为Nx2矩阵

% 构建系数矩阵a(x,y)（高斯型变系数）
C = 1 + a_amp * exp(-((X-x_0).^2/(2*c_x^2) + (Y-y_0).^2/(2*c_y^2)));
C = C(:);                   % 展平为向量

% 创建对角矩阵D（用于系数处理）
D = spdiags(C, 0, n^2, n^2);

%% ====================== RBF稀疏化源项 ========================
% 原始高斯源项
f_full = f_amp * exp(-((X-x_0).^2/(2*c_x^2) + (Y-y_0).^2/(2*c_y^2)));
f_full = f_full(:);

% 生成初始RBF中心点网格
[xk_x, xk_y] = meshgrid(linspace(0, 1, Nx_rbf), linspace(0, 1, Ny_rbf));
X_centers_rbf = [xk_x(:), xk_y(:)]; % 初始中心点

%% 交叉验证选择最优ε
cv_errors = zeros(size(epsilon_list));
for eps_idx = 1:length(epsilon_list)
    epsilon = epsilon_list(eps_idx);
    
    % 贪心算法初始化
    I = [];         % 选取的中心点索引
    A_rbf = [];     % RBF基函数矩阵
    r = f_full;     % 初始残差
    prev_cv_error = inf;
    
    % 贪心主循环
    for k = 1:min(100, Nx_rbf*Ny_rbf) % 限制最大迭代次数
        available = setdiff(1:size(X_centers_rbf,1), I);
        if isempty(available), break; end
        
        % 选择最优中心点
        max_inner = -inf;
        best_idx = 0;
        for idx = available
            dist = sqrt(sum((X_vec - X_centers_rbf(idx,:)).^2, 2));
            phi = exp(-epsilon*(dist).^2);
            inner = phi' * r;
            if abs(inner) > max_inner
                max_inner = abs(inner);
                best_idx = idx;
            end
        end
        
        % 更新基函数矩阵
        I = [I, best_idx];
        dist_new = sqrt(sum((X_vec - X_centers_rbf(best_idx,:)).^2, 2));
        A_rbf = [A_rbf, exp(-epsilon*(dist_new).^2)];
        
        % QR分解求解权重
        [Q, R] = qr(A_rbf, 0);
        w = R \ (Q' * f_full);
        r = f_full - A_rbf * w;
        
        % LOOCV误差计算
        H_ii = sum(Q.^2, 2);
        e_loo = r ./ (1 - H_ii);
        cv_error = mean(e_loo.^2);
        
        % 终止条件
        if cv_error > prev_cv_error || norm(r) < 1e-6
            break;
        end
        prev_cv_error = cv_error;
    end
    cv_errors(eps_idx) = cv_error;
end

% 选择最优ε
[~, opt_idx] = min(cv_errors);
epsilon_opt = epsilon_list(opt_idx);
fprintf('最优ε=%.2f\n', epsilon_opt);

%% 使用最优ε进行贪心稀疏化
I = []; A_rbf = []; r = f_full;
for k = 1: Nx_rbf*Ny_rbf
    available = setdiff(1:size(X_centers_rbf,1), I);
    if isempty(available), break; end
    
    % 选择最优中心点
    max_inner = -inf;
    best_idx = 0;
    for idx = available
        dist = sqrt(sum((X_vec - X_centers_rbf(idx,:)).^2, 2));
        phi = exp(-(epsilon_opt*dist).^2);
        inner = phi' * r;
        if abs(inner) > max_inner
            max_inner = abs(inner);
            best_idx = idx;
        end
    end
    
    % 更新基函数矩阵
    I = [I, best_idx];
    dist_new = sqrt(sum((X_vec - X_centers_rbf(best_idx,:)).^2, 2));
    A_rbf = [A_rbf, exp(-epsilon_opt*(dist_new).^2)];
    
    % QR分解求解权重
    [Q, R] = qr(A_rbf, 0);
    w = R \ (Q' * f_full);
    r = f_full - A_rbf * w;
    
    % 终止条件
    if norm(r) < 1e-6
        break;
    end

end

% 生成稀疏源项
f_sparse = A_rbf * w;

%% ====================== 方程求解 ========================
% 使用稀疏源项求解
b = D \ f_sparse;       % 注意这里使用稀疏源项
v = L \ (P*b);
w_solve = U \ v;
u_sparse = h^2 * w_solve;

% 使用完整源项求解（用于比较）
b_full = D \ f_full;
v_full = L \ (P*b_full);
w_full = U \ v_full;
u_full = h^2 * w_full;

%% ====================== 结果可视化 ========================
% 将解向量转换为网格格式
Z_full = zeros(n+2);
Z_sparse = zeros(n+2);
for i = 1:n
    for j = 1:n
        Z_full(i+1,j+1) = u_full(j+n*(i-1));
        Z_sparse(i+1,j+1) = u_sparse(j+n*(i-1));
    end
end

% 生成网格坐标
x1 = 0:h:1;
y1 = 0:h:1;

% 1. LOOCV误差曲线
figure;
semilogy(epsilon_list, cv_errors, 'bo-', 'LineWidth', 1.5);
xlabel('形状参数ε'); ylabel('LOOCV误差');
title('LOOCV误差随ε变化曲线');
grid on;

% 2. 源项对比
figure;
surf(X, Y, reshape(f_full, n, n));
title('完整源项 f_{full}'); colorbar;

figure;
surf(X, Y, reshape(f_sparse, n, n));
title(['稀疏源项 f_{sparse} (', num2str(numel(I)), '个中心点)']); 
colorbar;

% 3. 解场对比
figure;
surf(x1, y1, Z_full);
title('完整源项解场 u_{full}'); colorbar;

figure;
surf(x1, y1, Z_sparse);
title('稀疏源项解场 u_{sparse}'); colorbar;


% 误差分析
abs_error_u = abs(Z_full - Z_sparse);
max_error_u = max(abs_error_u(:));
rmse_u = sqrt(mean(abs_error_u(:).^2));
rmse_f = sqrt(mean((f_full - f_sparse).^2));

fprintf('源项最大绝对误差: %.2e\n', max(abs(f_full - f_sparse)));
fprintf('源项均方根误差: %.2e\n', rmse_f);
fprintf('解场最大绝对误差: %.2e\n', max_error_u);
fprintf('解场均方根误差: %.2e\n', rmse_u);
fprintf('使用中心点数: %d \n', numel(I));


