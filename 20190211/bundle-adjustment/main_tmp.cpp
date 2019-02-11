#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <iostream>
#include<opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
 #include <opencv2/line_descriptor.hpp>
 #include <opencv2/core/utility.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/features2d.hpp>
 #include <opencv2/highgui.hpp>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/features2d/features2d.hpp"
 #include <opencv2/line_descriptor.hpp>
 #include <opencv2/core/utility.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/features2d.hpp>
 #include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>  
#include "opencv2/opencv.hpp" 
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> 
#include<iostream>
#include"ORBMatching.h"
#include<g2o/core/sparse_optimizer.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/robust_kernel.h>
#include<g2o/core/robust_kernel_impl.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/cholmod/linear_solver_cholmod.h>
#include<g2o/types/slam3d/se3quat.h>
#include<g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
using namespace cv;
using namespace std;
using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace cv;
using namespace std;

using namespace std;
Mat MatchResult(Mat,Mat,double,int);
Mat KeyPointFind(Mat, double);
void graph_optimization(vector<Point2f> &pts1, vector<Point2f> &pts2, double fx, double fy, double cx, double cy) {
    cout << "--------bundle adjustment--------" << endl;

    //矩阵块：每个误差项优化变量维度为6，误差值维度为3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
    // Step1 选择一个线性方程求解器
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    // Step2 选择一个稀疏矩阵块求解器
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    // Step3 选择一个梯度下降方法，从GN、LM、DogLeg中选
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //添加两个位姿节点，默认值为单位Pose，并且将第一个位姿节点也就是第一帧的位姿固定
    for (int i = 0; i < 2; i++) {
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if (i == 0) {
            v->setFixed(true);
        }
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }

    //添加特征点作为节点
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        v->setId(2 + i);
        //深度未知，设为1，利用相机模型，相当于是在求归一化相机坐标
        double z = 1;
        double x = (pts1[i].x - cx) * z / fx;
        double y = (pts1[i].y - cy) * z / fy;
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
    }

    //相机内参
    g2o::CameraParameters *camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //添加第一帧中的边
    vector<g2o::EdgeProjectXYZ2UV *> edges;
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0)));

        edge->setMeasurement(Eigen::Vector2d(pts1[i].x, pts1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());

        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    //添加第二帧中的边
    for (size_t i = 0; i < pts2.size(); i++) {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1)));

        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    cout << "开始优化" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//结束计时
    cout << "优化完毕" << endl;

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    //输出变换矩阵T
    g2o::VertexSE3Expmap *v = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1));
    Eigen::Isometry3d pose = v->estimate();
    cout << "Pose=" << endl << pose.matrix() << endl;

    //优化后所有特征点的位置
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::VertexSBAPointXYZ *v = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2));
        cout << "vertex id:" << i + 2 << ",pos=";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0) << "," << pos(1) << "," << pos(2) << endl;
    }

    //估计inliner的个数
    int inliners = 0;
    for (auto e:edges) {
        e->computeError();
        //chi2()如果很大，说明此边的值与其它边很不相符
        if (e->chi2() > 1) {
            cout << "error = " << e->chi2() << endl;
        } else {
            inliners++;
        }
    }
    cout << "inliners in total points:" << inliners << "/" << pts1.size() + pts2.size() << endl;
    optimizer.save("ba.g2o");
    cout << "--------bundle adjustment--------" << endl;
    return;
};
int main(int argc, char **argv){
	double ratio = (double)atof(argv[1]);
	//ReadFile
	FILE *fin;
	fin=fopen("rgb.txt","rt");
	if(fin==NULL) {
	   printf("Fail To Open File rgb.txt!!");
	   return 0;
	}
	char s[10000][20];
	char s2[50]="rgb/";
	for(int i=0;i<10000;++i){
		fscanf(fin,"%s",&s[i]);
		//strcat(s2,s[i]);
		//strcat(s3,s2);
	}
		//printf("123456%s\n",s[0]);
	fclose(fin);
	int nImages= 1000;
	Mat image1;
	Mat image2;
//-------------------------------------------------------------------------------------------------------------------------------------------------------
	//Main Loop
	cout<<endl<<"Hello99"<<endl;
	Mat t_f,R_f;
	Mat traj = Mat::zeros(600, 600, CV_8UC3);
	for(int j=0;j<nImages;j++){
		//Divide ORB Match
		image1 = imread(s[j], 1);
		image2 = imread(s[j+1], 1);
//--------------------------------------------------------------------------------------------------------------------------------------------------------
	        //Cut The Photos
    		int offset_x = 0;
    		int offset_y = 0;
		cout<<endl<<"Hello0"<<endl;
    		cv::Rect roi;
    		roi.x = offset_x;
    		roi.y = offset_y;
    		roi.width = image1.size().width/4;
    		roi.height = image1.size().height/3;
    		cv::Mat crop1_1 = image1(roi);
    		roi.x = image1.size().width/4;
    		roi.y = 0;
   		 cv::Mat crop1_2 = image1(roi);
    		roi.x = (image1.size().width*2)/4;
    		roi.y = 0;
    		cv::Mat crop1_3 = image1(roi);
    		roi.x = (image1.size().width*3)/4;
    		roi.y = 0;
    		cv::Mat crop1_4 = image1(roi);
    		roi.x = 0;
    		roi.y = image1.size().height/3;
    		cv::Mat crop1_5 = image1(roi);
    		roi.x = image1.size().width/4;
    		roi.y = image1.size().height/3;
    		cv::Mat crop1_6 = image1(roi);
    		roi.x = (image1.size().width*2)/4;
    		roi.y = image1.size().height/3;
    		cv::Mat crop1_7 = image1(roi);
    		roi.x = (image1.size().width*3)/4;
    		roi.y = image1.size().height/3;
    		cv::Mat crop1_8 = image1(roi);
    		roi.x = 0;
    		roi.y = (image1.size().height*2)/3;
    		cv::Mat crop1_9 = image1(roi);
    		roi.x = image1.size().width/4;
    		roi.y = (image1.size().height*2)/3;
    		cv::Mat crop1_10 = image1(roi);
    		roi.x = (image1.size().width*2)/4;
    		roi.y = (image1.size().height*2)/3;
    		cv::Mat crop1_11 = image1(roi);
    		roi.x = (image1.size().width*3)/4;
    		roi.y = (image1.size().height*2)/3;
    		cv::Mat crop1_12 = image1(roi);
		//Cut Photo 2
    		//cv::Rect roi;
   		roi.x = offset_x;
    		roi.y = offset_y;
    		roi.width = image2.size().width/4;
    		roi.height = image2.size().height/3;
    		cv::Mat crop2_1 = image2(roi);
    		roi.x = image2.size().width/4;
    		roi.y = 0;
    		cv::Mat crop2_2 = image2(roi);
    		roi.x = (image2.size().width*2)/4;
    		roi.y = 0;
    		cv::Mat crop2_3 = image2(roi);
    		roi.x = (image2.size().width*3)/4;
    		roi.y = 0;
    		cv::Mat crop2_4 = image2(roi);
    		roi.x = 0;
    		roi.y = image2.size().height/3;
    		cv::Mat crop2_5 = image2(roi);
    		roi.x = image2.size().width/4;
    		roi.y = image2.size().height/3;
    		cv::Mat crop2_6 = image2(roi);
    		roi.x = (image2.size().width*2)/4;
    		roi.y = image2.size().height/3;
    		cv::Mat crop2_7 = image2(roi);
    		roi.x = (image2.size().width*3)/4;
    		roi.y = image2.size().height/3;
    		cv::Mat crop2_8 = image2(roi);
    		roi.x = 0;
    		roi.y = (image2.size().height*2)/3;
    		cv::Mat crop2_9 = image2(roi);
    		roi.x = image2.size().width/4;
    		roi.y = (image2.size().height*2)/3;
    		cv::Mat crop2_10 = image2(roi);
    		roi.x = (image2.size().width*2)/4;
    		roi.y = (image2.size().height*2)/3;
    		cv::Mat crop2_11 = image2(roi);
    		roi.x = (image2.size().width*3)/4;
    		roi.y = (image2.size().height*2)/3;
    		cv::Mat crop2_12 = image2(roi);
		cout<<endl<<"Hello1"<<endl;
		//Start to Get Feature points Result first photo
    		Mat FP1 = KeyPointFind(crop1_1,ratio);
    		Mat FP2 = KeyPointFind(crop1_2,ratio);
    		Mat FP3 = KeyPointFind(crop1_3,ratio);
    		Mat FP4 = KeyPointFind(crop1_4,ratio);
    		Mat FP5 = KeyPointFind(crop1_5,ratio);
    		Mat FP6 = KeyPointFind(crop1_6,ratio);
    		Mat FP7 = KeyPointFind(crop1_7,ratio);
    		Mat FP8 = KeyPointFind(crop1_8,ratio);
    		Mat FP9 = KeyPointFind(crop1_9,ratio);
    		Mat FP10 = KeyPointFind(crop1_10,ratio);
    		Mat FP11 = KeyPointFind(crop1_11,ratio);
    		Mat FP12 = KeyPointFind(crop1_12,ratio);
    		Mat origin1 = KeyPointFind(image1,ratio);
		cout<<endl<<"Hello101"<<endl;
    		// Get dimension of final image
    		int rows = FP1.cols*3;
    		int cols = FP1.rows*4;
		//Start to Get Feature points Result second photo
    		Mat SP1 = KeyPointFind(crop2_1,ratio);
    		Mat SP2 = KeyPointFind(crop2_2,ratio);
   		Mat SP3 = KeyPointFind(crop2_3,ratio);
    		Mat SP4 = KeyPointFind(crop2_4,ratio);
    		Mat SP5 = KeyPointFind(crop2_5,ratio);
    		Mat SP6 = KeyPointFind(crop2_6,ratio);
    		Mat SP7 = KeyPointFind(crop2_7,ratio);
    		Mat SP8 = KeyPointFind(crop2_8,ratio);
    		Mat SP9 = KeyPointFind(crop2_9,ratio);
    		Mat SP10 = KeyPointFind(crop2_10,ratio);
    		Mat SP11 = KeyPointFind(crop2_11,ratio);
    		Mat SP12 = KeyPointFind(crop2_12,ratio);
    		Mat origin2 = KeyPointFind(image2,ratio);
		cout<<endl<<"Hello102"<<endl;
		//Start To Match
	    	Mat GoodMatch1 = MatchResult(crop1_1,crop2_1,ratio,1);
	    	Mat GoodMatch2 = MatchResult(crop1_2,crop2_2,ratio,2);
	    	Mat GoodMatch3 = MatchResult(crop1_3,crop2_3,ratio,3);
	    	Mat GoodMatch4 = MatchResult(crop1_4,crop2_4,ratio,4);
	    	Mat GoodMatch5 = MatchResult(crop1_5,crop2_5,ratio,5);
	    	Mat GoodMatch6 = MatchResult(crop1_6,crop2_6,ratio,6);
	    	Mat GoodMatch7 = MatchResult(crop1_7,crop2_7,ratio,7);
	    	Mat GoodMatch8 = MatchResult(crop1_8,crop2_8,ratio,8);
	    	Mat GoodMatch9 = MatchResult(crop1_9,crop2_9,ratio,9);
	    	Mat GoodMatch10 = MatchResult(crop1_10,crop2_10,ratio,10);
	    	Mat GoodMatch11 = MatchResult(crop1_11,crop2_11,ratio,11);
	    	Mat GoodMatch12 = MatchResult(crop1_12,crop2_12,ratio,12);
	    	Mat GoodMatchOrigin = MatchResult(image1,image2,ratio,0);
		cout<<endl<<"Hello100"<<endl;
    		FILE *fp3;
    		fp3 = fopen("DivideORBMatchResult.txt","w+t");
    		std::ifstream infile1("rgb/tmp/MatchPoints1.txt");
		Mat DivideImage1Compare = imread(s[j], 1);
		Mat DivideImage2Compare = imread(s[j+1], 1);
		int a,b,c,d;
		cout<<endl<<"Hello2"<<endl;
   		while (infile1 >> a >> b>>c>>d){
         		circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
        		circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
	 		fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
         		fprintf(fp3,"\n");
    			// process pair (a,b)
    		}
    		std::ifstream infile2("rgb/tmp/MatchPoints2.txt");
    		while (infile2 >> a >> b>>c>>d){
         		circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
         		circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
         		fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
         		fprintf(fp3,"\n");
    			// process pair (a,b)
    		}
    		std::ifstream infile3("rgb/tmp/MatchPoints3.txt");
    		while (infile3 >> a >> b>>c>>d){
         		circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
         		circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
         		fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
         		fprintf(fp3,"\n");
    			// process pair (a,b)
    		}
		std::ifstream infile4("rgb/tmp/MatchPoints4.txt");
		while (infile4 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile5("rgb/tmp/MatchPoints5.txt");
		while (infile5 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile6("rgb/tmp/MatchPoints6.txt");
		while (infile6 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile7("rgb/tmp/MatchPoints7.txt");
		while (infile7 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile8("rgb/tmp/MatchPoints8.txt");
		while (infile8 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile9("rgb/tmp/MatchPoints9.txt");
		while (infile9 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile10("rgb/tmp/MatchPoints10.txt");
		while (infile10 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile11("rgb/tmp/MatchPoints11.txt");
		while (infile11 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		std::ifstream infile12("rgb/tmp/MatchPoints12.txt");
		while (infile12 >> a >> b>>c>>d){
			circle(DivideImage1Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			circle(DivideImage2Compare, Point(a,b), 4, Scalar(255,0,0), 1.5, 8, 0);
			fprintf(fp3,"%d\t%d\t%d\t%d",a,b,c,d);
			fprintf(fp3,"\n");
		    	// process pair (a,b)
		}
		fclose(fp3);
		cout<<endl<<"Hello3"<<endl;
		system("mv -f DivideORBMatchResult.txt rgb/tmp/");
		imshow("D1",DivideImage1Compare);
		//imshow("D2",DivideImage2Compare);
		waitKey(20);
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//StartToTrajectory
		string matpoint_name = "rgb/tmp/MatchPoints0.txt";	
		//string matpoint_name = "rgb/tmp/DivideORBMatchResult.txt";
  		// Set the camera calibration matrix
  		double f  = atof(argv[2]),
        	cx = atof(argv[3]), cy = atof(argv[4]);
		Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
		Matx33d newK;
		//Readline
		string s;
		int sTotal = 0;

		ifstream in;
		cout<<endl<<"Hello5"<<endl;
		in.open(matpoint_name);
		cout<<endl<<"Hello4"<<endl;
	
		while(!in.eof()) {
			getline(in, s);
			sTotal ++;	
		}
		cout <<endl<<"Number of Matching Points: "<<sTotal<<endl;
		in.close();	
		cout<<endl;
		const int npoints = sTotal; // number of point specified 

		// Points initialization. 
		// Only 2 ponts in this example, in real code they are read from file.

    		std::ifstream infile0(matpoint_name);
    		int a1, b1,c1,d1;
		Mat_<Point2f> points1t(1,npoints);
		Mat_<Point2f> points2t(1,npoints);
		int i=0;
    		while (infile0 >> a1 >> b1>>c1>>d1){	
			points1t(i) = Point2f(a1,b1);
			points2t(i) = Point2f(c1,d1);
			i++;
    		}
		vector<Point2f> points1,points2;
		points1 = points1t;
                points2 = points2t;
		graph_optimization(points1, points2, f, f,cx,cy);
		Mat F;
		vector<uchar> m_RANSACStatus;
		//Fundamental matrix  CV_FM_RANSAC
		Mat mask;
		F = findFundamentalMat(points1, points2, m_RANSACStatus, 4,3,0.1);
		cout<<endl<<"F = "<<endl<<F<<endl;
		//Normalized Fundamental Matrix START
		Mat newF;
		normalizeFundamental(F,newF);
		cout<<endl<<"newF = "<<endl<<newF<<endl;
		//Normalized Fundamental Matrix END
			Mat K2 = Mat::ones(3, 3, CV_64FC1);
			K2.at<double>(0, 0) = f;
			K2.at<double>(0, 1) = 0;
			K2.at<double>(0, 2) = cx;
			K2.at<double>(1, 0) = 0;
			K2.at<double>(1, 1) = f;
			K2.at<double>(1, 2) = cy;
			K2.at<double>(2, 0) = 0;
			K2.at<double>(2, 1) = 0;
			K2.at<double>(2, 2) = 1;
		//Essential from Fundamental
		Mat E = K2.t()*newF*K2;
		//Mat E;
		//essentialFromFundamental(newF,K,K,E);
		cout<<endl<<"Essential Matrix = "<<endl<<E<<endl;
		
		//2018/12/24
		
		cv::Point2d pp(cx,cy);
		//E = findEssentialMat(points2,points1,f,pp,RANSAC,0.999,1.0,mask);
		
		//MotionFromEssential
		//std::vector<Mat> Rs;
		//std::vector<Mat> ts;
		Mat Rs,ts;
		
		recoverPose(E, points2, points1, Rs, ts, f, pp);
		/*motionFromEssential(E,Rs,ts);
		vector<Mat>::iterator it = Rs.begin();
		cout<<endl<<"Show Rs: "<<endl;
		while (it != Rs.end())
		{
			Mat tmp = *it;
			cout<<endl<<tmp<<endl;
			it++;

		}
		vector<Mat>::iterator it2 = ts.begin();
		cout<<endl<<"Show ts: "<<endl;
		while (it2 != ts.end())
		{
			Mat tmp = *it2;
			cout<<endl<<tmp<<endl;
			it2++;

		}*/
		//ChooseRightSolution
		cout<<endl<<"Rotation : "<<Rs<<endl;
		cout<<endl<<"Translation : "<<ts<<endl;
		int scale = 1;
		if(j == 0){
			t_f = ts;
			R_f = Rs;
		}
		//Variable for map
		//double scale = 0.5;
		char text[100];
		int fontFace = FONT_HERSHEY_PLAIN;
		double fontScale = 1;
		int thickness = 1;
		cv::Point textOrg(10, 50);
		cout<<endl<<"Hello6"<<endl;
		/*if ((scale>0.1)&&(ts.at<double>(2)>ts.at<double>(0))&&(ts.at<double>(2)>ts.at<double>(1))){
			t_f = t_f + scale*(R_f*ts);
			R_f = Rs*R_f;
			cout<<endl<<"Change"<<endl;
		}*/
		if(ts.at<double>(2)>0){
			t_f = t_f + scale*(R_f*ts);
			R_f = Rs*R_f;
			cout<<endl<<"Change"<<endl;
		}
			/*t_f = t_f + scale*(R_f*ts);
			R_f = Rs*R_f;
			cout<<endl<<"Change"<<endl;*/
		cout<<endl<<"Rotation : "<<R_f<<endl;
		cout<<endl<<"Translation : "<<t_f<<endl;
		cout<<endl<<"Hello9"<<endl;
		int x = int(t_f.at<double>(0)) + 300;
		int y = int(t_f.at<double>(2)) + 100;
		cout<<endl<<"Hello8"<<endl;
		cout<<"x = "<<x<<" y = "<<y<<endl;
		circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);
		cout<<endl<<"Hello7"<<endl;
		//rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
		//void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
		//sprintf_s(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		//putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
		imshow("Trajectory", traj);
		waitKey(20);
	}

}
Mat KeyPointFind(Mat image1, double ratio){
		Ptr<FeatureDetector> detector = ORB::create(1000,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20 );
		Ptr<DescriptorExtractor> extractor = ORB::create(1000,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20 );
    
		// TODO default is 500 keypoints..but we can change
		//detector = FeatureDetector::create("ORB");  
		//extractor = DescriptorExtractor::create("ORB");
		//Ptr<descriptorextractor> extractor = ORB::create();
		//Ptr<featuredetector> detector = ORB::create(); 


		vector<KeyPoint> keypoints1, keypoints2;
		detector->detect(image1, keypoints1);

		cout << "# keypoints of image1 :" << keypoints1.size() << endl;
   
		Mat descriptors1,descriptors2;
		extractor->compute(image1,keypoints1,descriptors1);
                Mat img_keypoints_1;
                drawKeypoints( image1, keypoints1, img_keypoints_1, Scalar(255,0,0), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                return img_keypoints_1;
}
Mat MatchResult(Mat image1,Mat image2,double ratio,int index){
		Ptr<FeatureDetector> detector = ORB::create();
		Ptr<DescriptorExtractor> extractor = ORB::create();
    
		// TODO default is 500 keypoints..but we can change
		//detector = FeatureDetector::create("ORB");  
		//extractor = DescriptorExtractor::create("ORB");
		//Ptr<descriptorextractor> extractor = ORB::create();
		//Ptr<featuredetector> detector = ORB::create(); 

		clock_t begin = clock();

		vector<KeyPoint> keypoints1, keypoints2;
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);

		cout << "# keypoints of image1 :" << keypoints1.size() << endl;
		cout << "# keypoints of image2 :" << keypoints2.size() << endl;
   
		Mat descriptors1,descriptors2;
		extractor->compute(image1,keypoints1,descriptors1);
		extractor->compute(image2,keypoints2,descriptors2);

    

		cout << "Descriptors size :" << descriptors1.cols << ":"<< descriptors1.rows << endl;

		vector< vector<DMatch> > matches12, matches21;
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		matcher->knnMatch( descriptors1, descriptors2, matches12, 2 );
		matcher->knnMatch( descriptors2, descriptors1, matches21, 2 );
    
		// BFMatcher bfmatcher(NORM_L2, true);
		// vector<DMatch> matches;
		// bfmatcher.match(descriptors1, descriptors2, matches);
		cout << "Matches1-2:" << matches12.size() << endl;
		cout << "Matches2-1:" << matches21.size() << endl;

		// ratio test proposed by David Lowe paper = 0.8
		std::vector<DMatch> good_matches1, good_matches2;

		// Yes , the code here is redundant, it is easy to reconstruct it ....
		for(int i=0; i < matches12.size(); i++){
			if(matches12[i][0].distance < ratio * matches12[i][1].distance)
				good_matches1.push_back(matches12[i][0]);
		}

		for(int i=0; i < matches21.size(); i++){
			if(matches21[i][0].distance < ratio * matches21[i][1].distance)
				good_matches2.push_back(matches21[i][0]);
		}

		cout << "Good matches1:" << good_matches1.size() << endl;
		cout << "Good matches2:" << good_matches2.size() << endl;

		// Symmetric Test
		std::vector<DMatch> better_matches;
		for(int i=0; i<good_matches1.size(); i++){
			for(int j=0; j<good_matches2.size(); j++){
				if(good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx){
					better_matches.push_back(DMatch(good_matches1[i].queryIdx, good_matches1[i].trainIdx, good_matches1[i].distance));
					break;
				}
			}
		}
		
                    cout<<"better_matches :"<<better_matches.size()<<endl;

stringstream ss;
ss << better_matches.size();
string str = ss.str();
		    Mat img_matches;
    		    drawMatches( image1, keypoints1, image2, keypoints2, better_matches, img_matches, Scalar(255,0,0),
                    Scalar(255,0,0), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                    Mat img_keypoints_1;
                    drawKeypoints( image1, keypoints1, img_keypoints_1, Scalar(255,0,0), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		    string str1 ="Number: " +str;
                    putText(img_matches, str1, Point(30,70), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0,0,0), 1.5, 5);
    		    //-- Show detected matches
		    //imwrite("GoodMatch.png",img_matches);
    	            //imshow("Good Matches", img_matches );
    	            //waitKey();
// 0 是第幾張照片, 1 bettermatch的size, 2 Match 序號, 3 基準圖x座標,  4 基準圖y座標, 5 比較圖x座標, 6 比較圖y座標, 7所有人都有
		if(index == 0){
			FILE *fp2;
			fp2 = fopen("MatchPoints0.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints0.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x;
				int my = keypoints1[idx1].pt.y;
				int sx = keypoints2[idx1].pt.x;
				int sy = keypoints2[idx1].pt.y;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints0.txt rgb/tmp/");
                        //system("mkdir 123");
		}
		if(index == 1){
			FILE *fp2;
			fp2 = fopen("MatchPoints1.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints1.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x;
				int my = keypoints1[idx1].pt.y;
				int sx = keypoints2[idx1].pt.x;
				int sy = keypoints2[idx1].pt.y;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints1.txt rgb/tmp/");
		}
		if(index == 2){
			FILE *fp2;
			fp2 = fopen("MatchPoints2.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints2.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+160;
				int my = keypoints1[idx1].pt.y;
				int sx = keypoints2[idx1].pt.x+160;
				int sy = keypoints2[idx1].pt.y;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints2.txt rgb/tmp/");
		}
		if(index == 3){
			FILE *fp2;
			fp2 = fopen("MatchPoints3.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints3.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+320;
				int my = keypoints1[idx1].pt.y;
				int sx = keypoints2[idx1].pt.x+320;
				int sy = keypoints2[idx1].pt.y;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints3.txt rgb/tmp/");
		}
		if(index == 4){
			FILE *fp2;
			fp2 = fopen("MatchPoints4.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints4.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+480;
				int my = keypoints1[idx1].pt.y;
				int sx = keypoints2[idx1].pt.x+480;
				int sy = keypoints2[idx1].pt.y;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints4.txt rgb/tmp/");
		}
		if(index == 5){
			FILE *fp2;
			fp2 = fopen("MatchPoints5.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints5.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x;
				int my = keypoints1[idx1].pt.y+160;
				int sx = keypoints2[idx1].pt.x;
				int sy = keypoints2[idx1].pt.y+160;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints5.txt rgb/tmp/");
		}
		if(index == 6){
			FILE *fp2;
			fp2 = fopen("MatchPoints6.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints6.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+160;
				int my = keypoints1[idx1].pt.y+160;
				int sx = keypoints2[idx1].pt.x+160;
				int sy = keypoints2[idx1].pt.y+160;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints6.txt rgb/tmp/");
		}
		if(index == 7){
			FILE *fp2;
			fp2 = fopen("MatchPoints7.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints7.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+320;
				int my = keypoints1[idx1].pt.y+160;
				int sx = keypoints2[idx1].pt.x+320;
				int sy = keypoints2[idx1].pt.y+160;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints7.txt rgb/tmp/");
		}
		if(index == 8){
			FILE *fp2;
			fp2 = fopen("MatchPoints8.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints8.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+480;
				int my = keypoints1[idx1].pt.y+160;
				int sx = keypoints2[idx1].pt.x+480;
				int sy = keypoints2[idx1].pt.y+160;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints8.txt rgb/tmp/");
		}
		if(index == 9){
			FILE *fp2;
			fp2 = fopen("MatchPoints9.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints9.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x;
				int my = keypoints1[idx1].pt.y+320;
				int sx = keypoints2[idx1].pt.x;
				int sy = keypoints2[idx1].pt.y+320;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints9.txt rgb/tmp/");
		}
		if(index == 10){
			FILE *fp2;
			fp2 = fopen("MatchPoints10.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints10.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+160;
				int my = keypoints1[idx1].pt.y+320;
				int sx = keypoints2[idx1].pt.x+160;
				int sy = keypoints2[idx1].pt.y+320;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints10.txt rgb/tmp/");
		}
		if(index == 11){
			FILE *fp2;
			fp2 = fopen("MatchPoints11.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints11.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+320;
				int my = keypoints1[idx1].pt.y+320;
				int sx = keypoints2[idx1].pt.x+320;
				int sy = keypoints2[idx1].pt.y+320;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints11.txt rgb/tmp/");
		}
		if(index == 12){
			FILE *fp2;
			fp2 = fopen("MatchPoints12.txt","w+t");
			fclose(fp2);
			fp2 = fopen("MatchPoints12.txt","w+t");
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				int mx = keypoints1[idx1].pt.x+480;
				int my = keypoints1[idx1].pt.y+320;
				int sx = keypoints2[idx1].pt.x+480;
				int sy = keypoints2[idx1].pt.y+320;
		                fprintf(fp2,"%d\t%d\t%d\t%d",mx,my,sx,sy);
                                fprintf(fp2,"\n");
			}
			fclose(fp2);
                        system("mv -f MatchPoints12.txt rgb/tmp/");
		}


		    return img_matches;

}


