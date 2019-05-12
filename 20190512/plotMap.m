data = load("mvpMapPoints.txt");
x = data(:,3)*100+100;
y = data(:,4)*75+100;
z = data(:,5);
%plot3(x,y,z,'ro','linewidth',3);
plot(x,y,'ro','linewidth',3);
axis equal
