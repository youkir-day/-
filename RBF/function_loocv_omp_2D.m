clear;
clc;
%相关参数
n=50;
Nx=30;     
Ny=30;
%生成初始中心点
[xk_x,xk_y]=meshgrid(linspace(0,1,Nx),linspace(0,1,Ny));
X_centers =[xk_x(:),xk_y(:)]; %转换为(Nx*Ny)x2的中心坐标
%评估点
[xe_x,xe_y]=meshgrid(linspace(0,1,n),linspace(0,1,n));
X_test=[xe_x(:),xe_y(:)];
M=Nx*Ny;       %最多选择的基函数个数

%定义函数
F=@(X) function_F2(X);
y = F(X_test);   %真实的函数值
epsilon=10;    %形状参数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

%% 贪心算法稀疏化RBF
I=[];      %选取的中心点的索引
A=[];      %选择的基函数
r=y;       %初始残差
C=[];      %选取的中心点

prev_cv_error=inf;  %初始化LOOCV误差，初始为无穷大

%开始循环
for k=1:M
    available = setdiff(1:size(X_centers,1),I);  %未选择的索引
    if isempty(available)  %全选完了就停
        break;
    end
    %计算基函数与残差的内积
    max_inner=-inf;   %最大内积，初始为无穷小
    best_idx=0;       %最佳中心点的索引

    for idx = available
        %欧几里得距离
        dist = sqrt(sum((X_test - X_centers(idx,:)).^2, 2)); 
        phi=exp(-epsilon*(dist).^2);         %高斯径向基函数
        %计算内积
        inner = phi'* r;

        %更新最大内积和索引
        if abs(inner)>max_inner
            max_inner=abs(inner);
            best_idx=idx;
        end
    
    end

    %更新
    I=[I,best_idx];
    C=[C;X_centers(best_idx,:)];
    %新选点对应的基函数
    dist_new=sqrt(sum((X_test - X_centers(best_idx,:)).^2,2));%欧氏距离
    A=[A,exp(-epsilon*(dist_new).^2)];
    %更新w
    [Q,R]=qr(A,0);
    w = R \ (Q'*y);
    %更新残差
    r=y-A * w;


    %% 计算LOOCV误差
    H_ii=sum(Q.^2,2);
    denominator=1-H_ii; %分母
    e_loo=r./denominator;
    cv_error=mean(e_loo.^2);    %LOOCV误差，做终止条件
    %% 终止条件
   if ~isinf(prev_cv_error) && (cv_error > prev_cv_error)
        fprintf('LOOCV误差上升，停止于迭代%d，当前误差：%.2e，前次误差：%.2e\n', k, cv_error, prev_cv_error);
        break;
    end
    prev_cv_error=cv_error;  %更新前次误差
     % 残差终止条件：残差范数小于阈值时停止
   

end

%% 可视化
fe = A * w;

ZZ_true = reshape(y, [n, n]);
ZZ_pred = reshape(fe, [n, n]);

figure;
subplot(1,2,1);
surf(xe_x, xe_y, ZZ_true, 'EdgeColor', 'none'); % 真实函数曲面
title('真实函数');
xlabel('x1'); ylabel('x2'); zlabel('F');

subplot(1,2,2);
surf(xe_x, xe_y, ZZ_pred, 'EdgeColor', 'none'); % RBF近似曲面
title('RBF近似');
xlabel('x1'); ylabel('x2'); zlabel('F');

max_error = max(abs(y - fe));               % 最大绝对误差
rmse = sqrt(mean((y - fe).^2));             % 均方根误差
fprintf('最大绝对误差: %.2e\n', max_error);
fprintf('均方根误差 (RMSE): %.2e\n', rmse);
fprintf('使用中心数: %d\n', length(C));