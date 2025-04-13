function [epsilon_opt, cv_errors] = optimize_epsilon(X_test, f_full, Nx_rbf, Ny_rbf, epsilon_range)
    % LOOCV优化RBF形状参数epsilon
    % 输入:
    %   X_test      : 测试点坐标 (Nx2矩阵)
    %   f_full      : 完整源项向量 (Nx1)
    %   Nx_rbf, Ny_rbf : 初始中心点网格分辨率
    %   epsilon_range: 候选epsilon范围 [min, max, num_points]
    % 输出:
    %   epsilon_opt : 最优形状参数
    %   cv_errors    : LOOCV误差曲线

    %% 生成初始中心点网格
    [xk_x, xk_y] = meshgrid(linspace(0, 1, Nx_rbf), linspace(0, 1, Ny_rbf));
    X_centers = [xk_x(:), xk_y(:)];
    
    %% 计算距离矩阵
    DM = pdist2(X_test, X_centers);
    
    %% 定义RBF函数
    rbf = @(ep, r) exp(-(ep*r).^2);
    
    %% 生成候选epsilon列表
    epsilon_list = linspace(epsilon_range(1), epsilon_range(2), epsilon_range(3));
    cv_errors = zeros(size(epsilon_list));
    
    %% LOOCV主循环
    for i = 1:length(epsilon_list)
        ep = epsilon_list(i);
        Phi = rbf(ep, DM);
        
        % 计算帽子矩阵
        H = Phi * (Phi \ eye(size(Phi,1)));
        H_ii = diag(H);
        
        % 计算LOOCV残差
        residual_loo = (f_full - Phi*(Phi\f_full)) ./ (1 - H_ii);
        cv_errors(i) = mean(residual_loo.^2);
    end
    
    %% 寻找最优epsilon
    [~, idx] = min(cv_errors);
    epsilon_opt = epsilon_list(idx);
end
