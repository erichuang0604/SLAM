data = load("visual_odometry_epipolar.xyz");
x = data(:,1);
y = data(:,2);
z = data(:,3);
plot3(x,y,z,'ro','linewidth',3);
figure
data = load("groundtruth.txt");
x = data(:,1);
y = data(:,2);
z = data(:,3);
plot3(x,y,z,'ro','linewidth',3);
