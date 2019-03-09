#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <iostream>
#include <dirent.h>
#include <ctime>
#include <System.h>
using namespace cv;
using namespace std;
int main(int argc, const char *argv[]){
	Mat Image1 = cv::imread("5_14Test/00407.png", CV_LOAD_IMAGE_UNCHANGED);
	Mat Image2 = cv::imread("5_14Test/00466.png", CV_LOAD_IMAGE_UNCHANGED);
	FileStorage fs1("5_14Test/407.yml", FileStorage::READ);
	FileStorage fs2("5_14Test/466.yml", FileStorage::READ);
	FileNode kptFile1 = fs1["keypoints407"];
	FileNode dstFile1 = fs1["descriptors407"];
	FileNode kptFile2 = fs2["keypoints466"];
	FileNode dstFile2 = fs2["descriptors466"];
	cout<<"21"<<endl;	

	vector<KeyPoint> kpts1,kpts2;
        Mat dst1,dst2;
	read(kptFile1, kpts1);
	read(dstFile1, dst1);
	read(kptFile2, kpts2);
	read(dstFile2, dst2);
	cout<<"29"<<endl;
	cv::drawKeypoints(Image1, kpts1,Image1, Scalar::all(-1));
        //imshow("Keypoints", Image1);
	



        vector< vector<DMatch> > matches12, matches21;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
	matcher->knnMatch( dst1, dst2, matches12, 2 );
        matcher->knnMatch( dst2, dst1, matches21, 2 );
        // ratio test proposed by David Lowe paper = 0.8
        std::vector<DMatch> good_matches1, good_matches2;
        double ratio = 0.8;
        // Yes , the code here is redundant, it is easy to reconstruct it ....
        for(int i=0; i < matches12.size(); i++){
        	if(matches12[i][0].distance < ratio * matches12[i][1].distance)
                                good_matches1.push_back(matches12[i][0]);
        }
        for(int i=0; i < matches21.size(); i++){
                 if(matches21[i][0].distance < ratio * matches21[i][1].distance)
                                good_matches2.push_back(matches21[i][0]);
        }
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
	cout<<"56"<<endl;
	Mat output;
        drawMatches(Image1, kpts1, Image2, kpts2, better_matches, output);
	cout<<"58"<<endl;
        imshow("Matches result",output);
	cout<<"61"<<endl;
        waitKey(0);
		
}
