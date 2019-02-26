clear
%以下作面积-价格的散点图
A = xlsread('E:\pythonStudy\MachineLearning\房屋预测\Training_Set.xlsx');
Size = A(:,1);
Price = A(:,2);
plot(Size,Price,'*');
xlabel('Size');
ylabel('Price');
title('房价预测样本');
%以下作代价函数的曲线图
[m,n] = size(A);
[theta0,theta1]=meshgrid(-10000:10:10000,-10000:10:10000);
%J = symsum((theta0+theta1.*Size(i)-Price(i)).^2,1,m)
%J = J/(2.*m);
J = 0;
for i=1:m
    J = J+(theta0+theta1.*Size(i)-Price(i)).^2;
end
J = J./(2.*m)
mesh(theta0,theta1,J);
figure
surf(theta0,theta1,J)
%平滑处理,去除网格;
shading interp
