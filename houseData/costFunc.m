clear
%以下作面积-价格的散点图
A = xlsread('E:\pythonStudy\MachineLearning\房屋预测数据matlab表示\Training_Set.xlsx');
Size = A(:,1);
Price = A(:,2);
plot(Size,Price,'*');
xlabel('Size');
ylabel('Price');
title('房价预测样本');
%以下作代价函数的曲线图
[m,n] = size(A);
theta1 = [-15:0.5:15];
J = 0;
for i=1:m
    J = J+(theta1.*Size(i)-Price(i)).^2;
end
J = J./(2.*m);
plot(theta1,J);
xlabel('theta1');
ylabel('代价函数值');