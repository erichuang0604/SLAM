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
	//*************** read frames' name ***************//
	FILE *fin;
	fin = fopen("rgb.txt","rt");
	if(fin == NULL) {
		printf("Fail To Open File rgb.txt!!");
		return 0;
	}
	char s[10000][20];
	for(int i=0; i<10000; ++i){
		fscanf(fin, "%s" , &s[i]);
	}
	fclose(fin);
	int stop = 0;
	int total = 0;
	//*************** start to extract ORB-feature **************//
	int nImages = 10000;
	cv::Mat im;

	
	//Extract to the file
	FileStorage fs("keypoints.yml", FileStorage::WRITE);
	for(int ni = 0; ni < nImages; ni++){
		char s3[100] = "CompareRoom/";
	        char s4[100] = "keypoints";
       		char s5[100] = "descriptors";
		char number[20];
		sprintf(number, "%d", ni);
		strcat(s3,s[ni]);
		strcat(s4,number);
		strcat(s5,number);
		im = cv::imread(s3,CV_LOAD_IMAGE_UNCHANGED);
		//im = cv::imread("messi.jpg", CV_LOAD_IMAGE_UNCHANGED);
		if(im.empty()){
			cerr << endl << "Failed to load image at: "
		     	<< s3   << endl;
			stop = 1;
			total = ni;
		}
		if(stop == 1){
			write(fs, "NumberOfFrames",ni+1);
			ni = 10000;
		}
		Mat image1;
		image1 = im;
		Ptr<FeatureDetector> detector;
		Ptr<DescriptorExtractor> extractor;
        	extractor = DescriptorExtractor::create("ORB");
        	detector = new cv::OrbFeatureDetector(1000);
        	vector<KeyPoint> keypoints1;
        	detector->detect(image1, keypoints1);
        	Mat descriptors1;
        	extractor->compute(image1, keypoints1, descriptors1);
 		write( fs , s4, keypoints1);
		write( fs , s5, descriptors1);
		
	}
	fs.release();
}
