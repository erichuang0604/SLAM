#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <iostream>
#include <dirent.h>
#include <ctime>
#include <System.h>

#include <iostream>
#include <dirent.h>
#include <ctime>
using namespace cv;
using namespace std;


int main(int argc, const char *argv[]){

    if(argc != 4){
        cout << "usage:match <image1> <image2> <ratio>\n" ;
        exit(-1);
    }
  
    double ratio = (double)atof(argv[3]);
    string image1_name=string(argv[1]), image2_name = string(argv[2]);

    Mat image1 = imread(image1_name, 1);
    Mat image2 = imread(image2_name, 1);

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    
    // TODO default is 500 keypoints..but we can change
    detector = FeatureDetector::create("ORB");  
    extractor = DescriptorExtractor::create("ORB");

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

    cout << "Better matches:" << better_matches.size() << endl;
    
	//Print Match Points
    FILE *fp2;
    fp2 = fopen("OpencvORBMatches2.txt","w+t");
    for(int i = 0; i<better_matches.size(); i++){
        int idx1=better_matches[i].trainIdx;
        int idx2=better_matches[i].queryIdx;
    	fprintf(fp2,"%d\t%f\t%f\t%f\t%f\n",i,keypoints1[idx1].pt.x,keypoints1[idx1].pt.y,keypoints2[idx2].pt.x,keypoints2[idx2].pt.y);
    }
    fclose(fp2);


        //Change Coordinate, Normalized Camera, and Find Fundamental Matrix
	Mat points1(better_matches.size(),2,CV_32F);
	Mat points2(better_matches.size(),2,CV_32F);
	for(int i = 0; i<better_matches.size(); i++){
		int idx1=better_matches[i].trainIdx;
	        int idx2=better_matches[i].queryIdx;
		//points1.at<float>(i,0)=(keypoints1[idx1].pt.x-320)/448.651826741;
		//points1.at<float>(i,1)=(keypoints1[idx1].pt.y-240)/448.651826741;
		//points2.at<float>(i,0)=(keypoints2[idx2].pt.x-320)/517.34198667;
                //points2.at<float>(i,1)=(keypoints2[idx2].pt.y-240)/517.34198667;
		//points2.at<float>(i,0)=(keypoints2[idx2].pt.x-320)/448.651826741;
		//points2.at<float>(i,1)=(keypoints2[idx2].pt.y-240)/448.651826741;
		points1.at<float>(i,0)=keypoints1[idx1].pt.x;
		points1.at<float>(i,1)=keypoints1[idx1].pt.y;
		points2.at<float>(i,0)=keypoints2[idx2].pt.x;
                points2.at<float>(i,1)=keypoints2[idx2].pt.y;

	}

	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	m_Fundamental = findFundamentalMat(points1,points2,m_RANSACStatus,CV_FM_RANSAC);
	cout << "M = " << " "  << m_Fundamental << ";" << endl;
	
	//Get essential Matrix
	/*Mat K(3,3, CV_32FC2);
	K.at<float>(0,0) = 1;
	K.at<float>(0,1) = 0;
	K.at<float>(0,2) = 320;
	K.at<float>(1,0) = 0;
	K.at<float>(1,1) = 1;
	K.at<float>(1,2) = 240;
	K.at<float>(2,0) = 0;
	K.at<float>(2,1) = 0;
	K.at<float>(2,2) = 1;





	cout << "K = "<< endl << " "  << K << endl << endl;*/
	







    // show it on an image
    Mat output;
    drawMatches(image1, keypoints1, image2, keypoints2, better_matches, output);
    imshow("Matches result",output);
    waitKey(0);

    return 0;
    
}
