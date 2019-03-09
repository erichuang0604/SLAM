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

	if(argc !=2){
		cout<< "Please enter ratio you want!\n" ;
		exit(-1);
	}
	double ratio = (double)atof(argv[1]);
	//double elapsed_secs2 = 0;
	int stop = 0;
	int total = 0;

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
	//*************** start to extract ORB-feature **************//
	int nImages = 10000;
	cv::Mat im;
	//clock_t begin = clock();
	double elapsed_secs2 = 0;
	double elapsed_secs2_Extract = 0;
	double elapsed_secs2_Write = 0;

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
		if(stop == 1)
			ni = 10000;
		Mat image1;
		image1 = im;
		clock_t begin_Extract = clock();
		Ptr<FeatureDetector> detector;
		Ptr<DescriptorExtractor> extractor;
		detector = FeatureDetector::create("ORB");
		extractor = DescriptorExtractor::create("ORB");
	
		vector<KeyPoint> keypoints1;
		detector->detect(image1, keypoints1);
		//cout << "# keypoints of image1 :" << keypoints1.size() <<endl;
		Mat descriptors1;
		extractor->compute(image1, keypoints1, descriptors1);
		//cout << "Descriptors size :" << descriptors1.cols << ":"<< descriptors1.rows <<endl;
		clock_t end_Extract = clock();
		elapsed_secs2_Extract = elapsed_secs2_Extract + (double(end_Extract - begin_Extract) / CLOCKS_PER_SEC);
		clock_t begin_Write = clock();	
		cv::FileStorage store("template.bin", cv::FileStorage::APPEND);
		cv::write(store, s4 ,keypoints1);
		cv::write(store, s5 ,descriptors1);
		store.release();
		clock_t end_Write = clock();
		elapsed_secs2_Write = elapsed_secs2_Write + (double(end_Write - begin_Write) / CLOCKS_PER_SEC);
	}
	//clock_t end = clock();
	//double elapsed_secs2 = double(end - begin) / CLOCKS_PER_SEC;
	cout <<endl<< "Time Costs(Extract) : " << elapsed_secs2_Extract <<endl;
	cout <<"Average Time Costs(Extract) : " << elapsed_secs2_Extract/total <<endl;
	cout <<endl<< "Time Costs(Write) : " << elapsed_secs2_Write <<endl;
        cout <<"Average Time Costs(Write) : " << elapsed_secs2_Write/total <<endl;

	cout <<"Total: "<< total<< endl; 
	/*
	//Test whether the function is correct or not --> Part1
	vector<KeyPoint> keypoints2;
	Mat descriptors2;
	cv::FileStorage store2("template.bin", cv::FileStorage::READ);
	cv::FileNode n1 = store2["1Firstkeypoints"];
	cv::read(n1, keypoints2);
	cv::FileNode n2 = store2["Firstdescriptors"];
	cv::read(n2, descriptors2);
	store2.release();
	//Part2
	cv::FileStorage store3("template2.bin", cv::FileStorage::WRITE);
        cv::write(store3, "keypoints" ,keypoints2);
        cv::write(store3, "descriptors" ,descriptors2);
        store3.release();
	*/
				  		
}

























		
