clear
%���������-�۸��ɢ��ͼ
A = xlsread('E:\pythonStudy\MachineLearning\����Ԥ��\Training_Set.xlsx');
Size = A(:,1);
Price = A(:,2);
plot(Size,Price,'*');
xlabel('Size');
ylabel('Price');
title('����Ԥ������');
%���������ۺ���������ͼ
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
%ƽ������,ȥ������;
shading interp
