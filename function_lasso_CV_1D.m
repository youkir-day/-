clear;
clc;
%参数
N = 50;    
n = 100;
xk = linspace(0,1,N)';     %初始中心点
xe = linspace(0,1,n)';     %评估点
epsilon = 9;
%%
F=@(x)6*x.^2.*sin(12*x-4);
rbf=@(r)exp(-epsilon*(r).^2);
y=F(xe);

%构建评估矩阵
dist_matrix=pdist2(xe,xk);
A=rbf(dist_matrix);

%使用k折交叉验证，选择lambda
[W,FitInfo]=lasso(A,y,'CV',5);   %使用5折交叉验证

%选择最优lambda
lambda_option=FitInfo.Lambda1SE;
fprintf('最优lambda为：%.4f\n',lambda_option);

%使用最优lambda重新训练
w=W(:,FitInfo.Index1SE);    %对应权重
select_idx=find(w~=0);      %选出不为零的权重
select_xk=xk(select_idx);   %通过权重的索引，找出要选择的中心点，即选择的基函数
  
%构造稀疏模型
A_sparse=A(:,select_idx);
w_sparse=A_sparse\y;        %最小二乘优化权重
fe=A_sparse*w_sparse;

%误差分析
mse=mean((y-fe).^2);         %均方误差
error=max(abs(fe-y));        %最大绝对误差
fprintf('均方误差mse:%.2e\n最大绝对误差:%.2e\n',mse,error);
fprintf('使用基函数数量%d\n',length(select_xk));

%% 可视化
figure;
plot(xe,y,'k-','LineWidth',1.5);
hold on;
plot(xe,fe,'r--','LineWidth',1.5);
xlabel('x');
ylabel('f(x)');
title('使用交叉验证的lasso稀疏化RBF');
legend('真实解','lasso稀疏解');
grid on;