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
	system("rm -rf Examples/Monocular/template.bin");
        cv::FileStorage store0("template.bin", cv::FileStorage::WRITE);
        store0.release();
	
	//************* Read frames' name **************//
	FILE *fin;
	fin = fopen("rgb.txt","rt");
	if(fin == NULL){
		printf("Fail to open File rgb.txt\n");
		return 0;
	}
	char s[10000][20];
	for(int i = 0; i < 10000; ++i){
		fscanf(fin, "%s" , &s[i]);
	}
	fclose(fin);
	//*********** Start to extract ORB-feature *************//
	int nImages = 10000;
	cv::Mat im;
	for(int ni = 0; ni < nImages; ni++){
		char s1[100] = "CompareRoom/";
		char s2[100] = "keypoints";
		char s3[100] = "descriptors";
		char number[20];
		sprintf(number, "%d", ni);
		strcat(s1,s[ni]);
		strcat(s2, number);
		strcat(s3, number);
		im = cv::imread(s1,CV_LOAD_IMAGE_UNCHANGED);
		if(im.empty()){
			cerr << endl << "Failed to load image at: "<<s1<<endl;
			stop = 1;
			total = ni;
		}
		if(stop == 1)
			ni = 10000;
		Mat image1;
		image1 = im;
		Ptr<FeatureDetector> detector;
		Ptr<DescriptorExtractor> extrator;
		extractor = DescriptorExtractor::create("ORB");
	        detector = new cv::OrbFeatureDetector(100);
		vector<KeyPoint> keypoints1;
		detector->detect(image1, keypoints1);
		Mat descriptors1;
		extractor->compute(image1, keypoints1, descriptors1);
		
		cv::FileStorage store("BagOfFeature.bin", cv::FileStorage::APPEND);
		cv::write(store, s2, keypoints1);
		cv::write(store, s3, descriptors1);
		store.release();
	}		
}












