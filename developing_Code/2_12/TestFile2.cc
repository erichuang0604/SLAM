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
	FileStorage fs("keypoints.yml", FileStorage::WRITE);
	cv::Mat image1 = cv::imread("developing_Code/2_12/00536.png", CV_LOAD_IMAGE_UNCHANGED);
	Ptr<FeatureDetector> detector;
        Ptr<DescriptorExtractor> extrator;
        extrator = DescriptorExtractor::create("ORB");
        detector = new cv::OrbFeatureDetector(100);
        vector<KeyPoint> keypoints1;
        detector->detect(image1, keypoints1);
        Mat descriptors1;
        extrator->compute(image1, keypoints1, descriptors1);
	write( fs , "i00213k", keypoints1);
        write( fs , "i00213d", descriptors1);
	fs.release();




	
	FileStorage fs2("keypoints.yml", FileStorage::READ);
	vector<KeyPoint> kpts0;
        Mat dst0;
	FileNode kptFileNode0 = fs2["i00213k"];
        read( kptFileNode0, kpts0);
	FileNode kptFileNode00 = fs2["i00213d"];
        read( kptFileNode00, dst0 );


	Mat CompareImage = cv::imread("developing_Code/2_12/00536.png", CV_LOAD_IMAGE_UNCHANGED);
        Ptr<FeatureDetector> detectorCompare;
        Ptr<DescriptorExtractor> extractorCompare;
        detectorCompare = new cv::OrbFeatureDetector(100);   //nFeaturePoints 
        extractorCompare = DescriptorExtractor::create("ORB");
        vector<KeyPoint> keypointsCompare;
        detectorCompare -> detect(CompareImage, keypointsCompare);
        Mat descriptorsCompare;
        extractorCompare -> compute(CompareImage, keypointsCompare, descriptorsCompare);

	vector< vector<DMatch> > matches12, matches21;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
	matcher->knnMatch( descriptorsCompare, dst0, matches12, 2 );
        matcher->knnMatch( dst0, descriptorsCompare, matches21, 2 );

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

		cout<<endl<<" : "<<better_matches.size()<<endl;


}
