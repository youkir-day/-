 function F = function_F2(X)
    % 初始化输出向量
    F = zeros(size(X, 1), 1);
    
    % 高斯参数
    A1 = 10; A2 = 10;       % 幅值
    sigma_sq = 0.02;         % 方差参数
    center1 = [0.25, 0.25];  % 第一个高斯中心
    center2 = [0.75, 0.75];  % 第二个高斯中心
    
    for i = 1:size(X, 1)
        x = X(i, 1);
        y = X(i, 2);
        
        % 计算第一个高斯项
        term1 = A1 * exp(-((x - center1(1))^2 + (y - center1(2))^2) / sigma_sq);
        
        % 计算第二个高斯项
        term2 = A2 * exp(-((x - center2(1))^2 + (y - center2(2))^2) / sigma_sq);
        
        % 叠加结果
        F(i) = term1 + term2;
    end
end