clear; clc;

% 参数
N = 50;           % 中心点数
n =100;
xk = linspace(0, 1, N)';
xe = linspace(0,1,n)';
F=@(x) 6 * x.^2 .* sin(12*x - 4);
y = F(xe);
epsilon = 3;      % 形状参数
M = 20;           % 最大中心点数,最大基函数个数
tol = 1e-4;
loo_history=[];

% 基函数矩阵
Phi = exp(-epsilon * (pdist2(xe,xk)).^2);  

%% 贪心算法初始化
I = [];     % 已选中心索引
A = [];     % 设计矩阵
C = [];     % 中心点坐标
r = y;      % 初始残差

%% 带LOOCV的贪心迭代
for k = 1:M
    available = setdiff(1:N, I);
    if isempty(available)
        break; 
    end
    
    min_loo = inf;    %初始为无穷大
    best_col = -1;
    
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
     % 遍历计算LOO误差
   for j = available
        % A_temp = [A, Phi(:,j)];
        % [Q_temp, R_temp] = qr(A, 0);
        % w_temp = R_temp \ (Q_temp' * y);
        % r_temp = y - A * w_temp;
        H_ii = sum(Q.^2, 2);
        loo_e = r ./ (1 - H_ii);
        loo_mse = mean(loo_e.^2);
        
        if loo_mse < min_loo
            min_loo = loo_mse;
            best_indx = j;
        end
    end
    % 记录LOOCV误差历史
    loo_history = [loo_history, min_loo];
    %终止条件
     if length(loo_history) >= 2 && loo_history(end) > loo_history(end-1)
        fprintf('LOOCV误差上升，终止于迭代 %d\n', k);
        break;
     end
end

%% 预测函数
fe=A*w;

%% 可视化
yy_true = 6 * xe.^2 .* sin(12*xe - 4);
yy_pred = fe;

figure;
plot(xe, yy_true, 'k-', 'LineWidth', 1);
hold on;
plot(xe, yy_pred, 'r--', 'LineWidth', 1.5);
xlabel('x'); ylabel('y');
legend('真实函数', '稀疏RBF近似');
title('使用LOOCV的贪心算法稀疏化RBF近似');
grid on;

% 误差计算
error = max(abs(yy_true - yy_pred));
fprintf('最大绝对误差:%.2e\n',error);
fprintf('最小LOO_MSE:%.2e\n',min_loo);
disp(['使用的基函数数量: ', num2str(length(C))]);
