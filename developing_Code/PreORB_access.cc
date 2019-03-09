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
	if(argc !=4){
                cout<< "Please enter filename and ratio you want!\n" ;
		cout<< "And how many pictures you have!\n";
                exit(-1);
        }
	int nImages = (int)atof(argv[3]);
	double ratio = (double)atof(argv[2]);
	string image1_name = string(argv[1]);
	//********* Extract target image ************//
	Ptr<FeatureDetector> detector;       //2018_2_6_important part
        Ptr<DescriptorExtractor> extractor;
	//detector = FeatureDetector::create("ORB"); //2018_2_6_10:04
        extractor = DescriptorExtractor::create("ORB");

	detector = new cv::OrbFeatureDetector(100);
	//extractor = new cv::OrbDescriptorExtractor;  //2018_2_6_10:05






	
	Mat image1 = imread(image1_name, 1);
	
//	clock_t begin = clock();
        vector<KeyPoint> keypoints1;
        detector->detect(image1, keypoints1);
	Mat descriptors1;
        extractor->compute(image1,keypoints1,descriptors1);
 //	clock_t end = clock();
	int tmp = 0;
	int currentImg;
	double elapsed_secs2 = 0;
	double elapsed_secs = 0;
	double elapsed_secs_Read = 0;
	double elapsed_secs_Compare = 0;
//	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;	
	for(int ni = 0; ni < nImages; ni++){
		char s4[100] = "keypoints";
                char s5[100] = "descriptors";
		char number[20];
                sprintf(number, "%d", ni);
                strcat(s4,number);
                strcat(s5,number);

		clock_t begin_Read = clock();
		vector<KeyPoint> keypoints2;
		Mat descriptors2;
		cv::FileStorage store2("template.bin", cv::FileStorage::READ);
	        cv::FileNode n1 = store2[s4];
	        cv::read(n1, keypoints2);
	        cv::FileNode n2 = store2[s5];
	        cv::read(n2, descriptors2);
	        store2.release();
		clock_t end_Read = clock();
		elapsed_secs_Read = elapsed_secs_Read + (double(end_Read - begin_Read) / CLOCKS_PER_SEC);


	
		clock_t begin_Compare = clock();	
		vector< vector<DMatch> > matches12, matches21;
        	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
       		//Flann-based matcher
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");

		matcher->knnMatch( descriptors1, descriptors2, matches12, 2 );
        	matcher->knnMatch( descriptors2, descriptors1, matches21, 2 );
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
                //cout << "Good matches1:" << good_matches1.size() << endl;
                //cout << "Good matches2:" << good_matches2.size() << endl;
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
		clock_t end_Compare = clock();
		elapsed_secs_Compare = elapsed_secs_Compare + (double(end_Compare - begin_Compare) / CLOCKS_PER_SEC);
//		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		if(better_matches.size() > tmp){
                        tmp = better_matches.size();
                        currentImg = ni;
                }

	}
//	clock_t end = clock();
//        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//        elapsed_secs2 = elapsed_secs;


	cout << "Better matches:" << tmp << endl;
        cout << "Current Image is "<<currentImg <<endl;
        cout << "Time Costs(Read) : " << elapsed_secs_Read << endl;
	cout << "Average Costs(Read) : "<<elapsed_secs_Read/nImages <<endl;
	cout << "Time Costs(Compare) : "<< elapsed_secs_Compare <<endl;
	cout << "Average Costs(Compare) : "<<elapsed_secs_Compare/nImages <<endl;
}











