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
	
	//Extract to the file
	FileStorage fs("keypoints.yml", FileStorage::WRITE);
	for(int ni=0; ni < 5; ni++){
		Mat image1;
		if(ni == 0){
			image1 = cv::imread("developing_Code/2_12/Test/00213.png", CV_LOAD_IMAGE_UNCHANGED);
			//namedWindow("Display window0", WINDOW_AUTOSIZE); 
			//imshow("Display window0", image1);
			cout<<endl<<"22"<<endl;
		}
		if(ni == 1){
			image1 = cv::imread("developing_Code/2_12/Test/00286.png", CV_LOAD_IMAGE_UNCHANGED);
			//namedWindow("Display window1", WINDOW_AUTOSIZE); 
			//imshow("Display window1", image1);
                }
		if(ni == 2){
			image1 = cv::imread("developing_Code/2_12/Test/00536.png", CV_LOAD_IMAGE_UNCHANGED);
			//namedWindow("Display window2", WINDOW_AUTOSIZE); 
			//imshow("Display window2", image1);
                }
		if(ni == 3){
			image1 = cv::imread("developing_Code/2_12/Test/00572.png", CV_LOAD_IMAGE_UNCHANGED);
                }
		if(ni == 4){
			image1 = cv::imread("developing_Code/2_12/Test/00856.png", CV_LOAD_IMAGE_UNCHANGED);
                }
        	Ptr<FeatureDetector> detector;
        	Ptr<DescriptorExtractor> extrator;
        	extrator = DescriptorExtractor::create("ORB");
        	detector = new cv::OrbFeatureDetector(100);
        	vector<KeyPoint> keypoints1;
        	detector->detect(image1, keypoints1);
        	Mat descriptors1;
        	extrator->compute(image1, keypoints1, descriptors1);
			cout<<endl<<"48"<<endl;	
		if(ni == 0){
 			write( fs , "i00213k", keypoints1);
			write( fs , "i00213d", descriptors1);
		}
		if(ni == 1){
			write( fs , "i00286k", keypoints1);
                        write( fs , "i00286d", descriptors1);
		}
		if(ni == 2){
			write( fs , "i00536k", keypoints1);
                        write( fs , "i00536d", descriptors1);
		}
		if(ni == 3){
			write( fs , "i00572k", keypoints1);
                        write( fs , "i00572d", descriptors1);
		}
		if(ni == 4){
			write( fs , "i00856k", keypoints1);
                        write( fs , "i00856d", descriptors1);
		}
	}
	fs.release();
	
	//Read the file and Compare
	vector<KeyPoint> kpts0, kpts1, kpts2, kpts3, kpts4;
	Mat dst0, dst1, dst2, dst3, dst4;
  	FileStorage fs2("keypoints.yml", FileStorage::READ);
  	//
	FileNode kptFileNode0 = fs2["i00213k"];
  	read( kptFileNode0, kpts0);
	FileNode kptFileNode1 = fs2["i00286k"];
	read( kptFileNode1, kpts1);
	FileNode kptFileNode2 = fs2["i00536k"];
	read( kptFileNode2, kpts2);
	FileNode kptFileNode3 = fs2["i00572k"];
        read( kptFileNode3, kpts3);
	FileNode kptFileNode4 = fs2["i00856k"];
        read( kptFileNode4, kpts4);
	//
	FileNode kptFileNode00 = fs2["i00213d"];
        read( kptFileNode00, dst0 );
        FileNode kptFileNode11 = fs2["i00286d"];
        read( kptFileNode11, dst1 );
        FileNode kptFileNode22 = fs2["i00536d"];
        read( kptFileNode22, dst2 );
        FileNode kptFileNode33 = fs2["i00572d"];
        read( kptFileNode33, dst3 );
        FileNode kptFileNode44 = fs2["i00856d"];
        read( kptFileNode44, dst4 );
  	fs2.release();
		cout<<endl<<"99"<<endl;	
	//CompareImageExtract
	Mat CompareImage = cv::imread("developing_Code/2_12/Test/00536.png", CV_LOAD_IMAGE_UNCHANGED);
        Ptr<FeatureDetector> detectorCompare;
        Ptr<DescriptorExtractor> extractorCompare;
	detectorCompare = new cv::OrbFeatureDetector(100);   //nFeaturePoints 
        extractorCompare = DescriptorExtractor::create("ORB");
	vector<KeyPoint> keypointsCompare;
	detectorCompare -> detect(CompareImage, keypointsCompare);
	Mat descriptorsCompare;
	extractorCompare -> compute(CompareImage, keypointsCompare, descriptorsCompare);
		cout<<endl<<"110"<<endl;	
	//Compare
	for(int i=0; i<5; i++){
		vector< vector<DMatch> > matches12, matches21;
        	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
		cout<<endl<<"115"<<endl;
		if(i == 0){
        		matcher->knnMatch( descriptorsCompare, dst0, matches12, 2 );
        		matcher->knnMatch( dst0, descriptorsCompare, matches21, 2 );
		cout<<endl<<"119"<<endl;
		}
		if(i == 1){
                        matcher->knnMatch( descriptorsCompare, dst1, matches12, 2 );
                        matcher->knnMatch( dst1, descriptorsCompare, matches21, 2 );
		cout<<endl<<"124"<<endl;
                }
		if(i == 2){
                        matcher->knnMatch( descriptorsCompare, dst2, matches12, 2 );
                        matcher->knnMatch( dst2, descriptorsCompare, matches21, 2 );
                }
		if(i == 3){
                        matcher->knnMatch( descriptorsCompare, dst3, matches12, 2 );
                        matcher->knnMatch( dst3, descriptorsCompare, matches21, 2 );
                }
		if(i == 4){
                        matcher->knnMatch( descriptorsCompare, dst4, matches12, 2 );
                        matcher->knnMatch( dst4, descriptorsCompare, matches21, 2 );
                }

		cout<<endl<<"136"<<endl;
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
		cout<<endl<<i<<" : "<<better_matches.size()<<endl;
	}
	
	

}
