clear;
clc;
% 生成测试数据
N = 50;       %中心点数量
n =100;
xk = linspace(0, 1, N)';
xe = linspace(0,1,n)';
F=@(x) 6 * x.^2 .* sin(12*x - 4);
y = F(xe);
rbf=@(r) exp(-epsilon*(r).^2);
tol = 1e-3;   %阈值

% RBF参数
epsilon = 3;    % 形状参数
M = 25;         % 最大中心数

% 预先计算所有基函数矩阵（N x N）
Phi = exp(-epsilon *(pdist2( xe,xk)).^2);

% 贪心算法初始化
I = [];         % 选中中心索引
A = [];         % 设计矩阵
r = y;          % 初始残差
C = [];         % 中心集合

% 迭代选择中心
for k = 1:M
    available = setdiff(1:N, I);  % 未选中的候选索引
    if isempty(available)
        break;
    end
    
    % 计算内积并选择最大投影
    inner_products =  Phi(:, available)'*r;
    [~, idx] = max(abs(inner_products));
    best = available(idx);
   
    % 更新设计矩阵和中心集合
    I = [I, best];
    A = [A, Phi(:, best)];
    C = [C; xk(best)];
    
    % 最小二乘更新权重
     % QR分解求解权重 
    [Q, R] = qr(A, 0);  
    w = R \ (Q' * y);     
    
    % 更新残差
    r = y - A * w;
    current_error = norm(r);
    % --- 终止条件 ---
    if current_error < tol
        fprintf('收敛于迭代 %d，残差范数=%.2e\n', k, current_error);
        break;
    end
end

% 预测函数
fe=A*w;

% 可视化结果
yy_true = 6 * xe.^2 .* sin(12*xe - 4);
yy_pred =fe;

figure;
plot(xe, yy_true, 'k-', 'LineWidth', 1.5);
hold on;
plot(xe, yy_pred, 'r--', 'LineWidth', 1.5);
xlabel('x'); 
ylabel('y');
% legend( '真实函数', '稀疏RBF近似');
title('贪心稀疏化RBF近似结果');
grid on;

% 计算误差
error = max(abs(yy_true - yy_pred));
fprintf('最大绝对误差: %.2e\n',error)
disp(['基函数数量：',num2str(length(C))]);
