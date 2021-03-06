#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "ORBextractor.h"
#include <fstream> 
using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    //-- 讀取圖像
    Mat img_1 = imread ( "515.jpg" );
    Mat mImGray=img_1;
    Mat outimg1,outimg2;//輸出圖像
    cvtColor(mImGray,mImGray,CV_RGB2GRAY);//轉換爲灰度圖
    



    //opencv中接口函數
   std::vector<KeyPoint> keypoints_1,keypoints_2,keypoints_3;
    Mat descriptors_1,descriptors_2,descriptors_3;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    detector->detect ( mImGray,keypoints_1 );
    descriptor->compute ( mImGray, keypoints_1, descriptors_1 );
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("opencv提取ORB特徵點",outimg1);

    //調用ORB SLAM中特徵提取函數
    ORB_SLAM2::ORBextractor* mpIniORBextractor;
    mpIniORBextractor = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);//From TUM4
    (*mpIniORBextractor)(mImGray,cv::Mat(),keypoints_2,descriptors_2 ) ;
    drawKeypoints( img_1, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB SLAM提取ORB特徵點",outimg2);
cout<<"text2"<<endl;

// read:
FileStorage fs("/home/erichuang0604/ORB_SLAM2/rgbResult/descriptor/descriptors514.txt",FileStorage::READ);
fs["mat1"] >> descriptors_3;

// read:
FileStorage fs2("/home/erichuang0604/ORB_SLAM2/rgbResult/keypoint/keypoints514.txt",FileStorage::READ);
fs2["mat1"] >> keypoints_3;
cout<<"text"<<endl;










    //Matching
    waitKey(0);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );
   //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 1*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }
    //ORBmatcher matcher(0.9,true);


    // show it on an image
    Mat output;
    drawMatches(img_1, keypoints_1, img_1, keypoints_2, good_matches, output);
    imshow("Matches result",output);
    waitKey(0);


}
