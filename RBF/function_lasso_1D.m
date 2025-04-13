clear;
clc;
%参数
xk=linspace(0,1,50)';
xe=linspace(0,1,100)';
epsilon=3;
lambda=0.001;

F=@(x) 6*x.^2.*sin(12*x.^2-4);
y=F(xe);
rbf=@(r) exp(-epsilon*(r).^2);

%构建设计矩阵，初始基函数矩阵
dist_matrix =pdist2(xe,xk);  %xe-xk

A=rbf(dist_matrix);

%% 求解Lasso回归
[W,FitInfo]=lasso(A,y,'Lambda',lambda);   %使用内置的lasso函数求解
w=W(:,1);                           %
% select_index=find(abs(w)>1e-6);     %选择大于0的系数
select_index=find(w~=0);
select_xk =xk(select_index);

%% 得到稀疏化基函数矩阵
A_sparse=A(:,select_index);
%QR分解
[Q,R]=qr(A_sparse,0);
w_sparse=R\(Q'*y);

fe=A_sparse*w_sparse;       %近似函数

%% 误差分析
mse=mean((y-fe).^2);     %均方差
error=max(abs(y-fe));
fprintf('最大误差为：%.2e\n均方差为：%.2e\n',error,mse);
disp(['基函数数量:',num2str(length(w_sparse))]);

%%可视化
figure;
plot(xe,y,'k-','LineWidth',1.5);
hold on;
plot(xe,fe,'r--','LineWidth',2);
xlabel('x');
ylabel('y');
title('使用Lasso稀疏化RBF');
legend('真实值','Lasso稀疏RBF');
