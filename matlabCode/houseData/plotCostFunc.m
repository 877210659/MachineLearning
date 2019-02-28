clear
%���������-�۸��ɢ��ͼ
A = xlsread('E:\pythonStudy\MachineLearning\����Ԥ������matlab��ʾ\Training_Set.xlsx');
Size = A(:,1);
Price = A(:,2);
plot(Size,Price,'*');
xlabel('Size');
ylabel('Price');
title('����Ԥ������');
%���������ۺ���������ͼ
[m,n] = size(A);
theta1 = [-15:0.5:15];
J = 0;
for i=1:m
    J = J+(theta1.*Size(i)-Price(i)).^2;
end
J = J./(2.*m);
plot(theta1,J);
xlabel('theta1');
ylabel('���ۺ���ֵ');