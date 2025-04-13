function A = DiscretePoisson2D(n)
    A = zeros(n*n, n*n);  % 初始化n²×n²零矩阵

    % 主对角线赋值为4
    for i = 1:n*n
        A(i, i) = 4; 
    end

    % 横向邻居（块内次对角线）
    for k = 1:n         % 遍历每个块（共n个块）
        for i = 1:n-1   % 块内相邻节点
            row = n*(k-1) + i;    % 当前节点的一维索引
            A(row, row+1) = -1;   % 右邻居
            A(row+1, row) = -1;   % 左邻居（对称位置）
        end
    end

    % 纵向邻居（块间次对角线）
    for i = 1:n*(n-1)   % 遍历前n(n-1)个节点
        A(i, i+n) = -1; % 下方邻居
        A(i+n, i) = -1; % 上方邻居（对称位置）
    end
end
