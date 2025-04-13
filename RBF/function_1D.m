clear;
clc;
F = @(x) 6 * x.^2 .* sin(12*x - 4);
epsilon =3;
rbf=@(r)exp(-(epsilon*r).^2);

xk=linspace(0,1,50);  %中心点
xe=linspace(0,1,100);
fk=F(xk);
y=F(xe);
A=zeros(length(xk),length(xk));
for i=1:length(xk)
    for j =1:length(xk)
        A(i,j)=rbf(abs(xk(i)-xk(j)));
    end
end
w=A\fk';
B =zeros(length(xe),length(xk));
for i=1:length(xe)
    for j=1:length(xk)
        B(i,j)=rbf(abs(xe(i)-xk(j)));
    end
end
fe=B*w;          %近似解

figure;
plot(xe, y, 'b-', 'LineWidth', 1.5); 
hold on;
plot(xe, fe, 'r--', 'LineWidth', 1.2);
xlabel('x');
ylabel('f(x)');
legend('真实解', 'RBF插值');
title('RBF插值结果对比');

% 计算误差
error = max(abs(y - fe'));
fprintf('最大绝对误差:%.2e \n', error);
mse=mean((y-fe').^2);     %均方差
fprintf('均方误差:%.2e\n',mse);
disp(['使用基函数数量',num2str(length(xk))]);


