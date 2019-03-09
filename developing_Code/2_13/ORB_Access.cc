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
	//CompareImageExtract
	//Mat CompareImage = cv::imread("developing_Code/2_12/Test/00536.png", CV_LOAD_IMAGE_UNCHANGED);
	Mat CompareImage = cv::imread("00681.png", CV_LOAD_IMAGE_UNCHANGED);
	


        Ptr<FeatureDetector> detectorCompare;
        Ptr<DescriptorExtractor> extractorCompare;
	detectorCompare = new cv::OrbFeatureDetector(1000);   //nFeaturePoints 
        extractorCompare = DescriptorExtractor::create("ORB");
	vector<KeyPoint> keypointsCompare;
	detectorCompare -> detect(CompareImage, keypointsCompare);
	Mat descriptorsCompare;
	extractorCompare -> compute(CompareImage, keypointsCompare, descriptorsCompare);
	cv::drawKeypoints(CompareImage, keypointsCompare,CompareImage, Scalar::all(-1));
	imshow("Keypoints", CompareImage);	
	

	FILE *fp;
	fp = fopen("OpencvORBCompareImageFeaturePoint.txt","w+t");
	if(fp == NULL) {
		printf("Fail to Open File");
		return 0;
	}
	for(int i=0;i<1000;i++){
		float x = keypointsCompare[i].pt.x;
		float y = keypointsCompare[i].pt.y;
		fprintf(fp,"%f\t%f\n",x,y);
	}
	fclose(fp); 
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
	int n=0;
	FileStorage fs3("keypoints.yml", FileStorage::READ);
	fs3["NumberOfFrames"] >> n;
	n = (int) fs3["NumberOfFrames"];
	cout<<"Numbers of frames : "<<n;

	int tmp = 0;
	int currentImg;
	vector<KeyPoint> Bestkpts;
	std::vector<DMatch> best_matches;
	double Time_To_Read = 0;
	double Time_To_Compare = 0;
	double Time_To_Open_File = 0;
	double Time_To_Close_File = 0;
	clock_t begin_open = clock();
	FileStorage fs2("keypoints.yml", FileStorage::READ);
	clock_t end_open = clock();
	Time_To_Open_File = (double(end_open - begin_open) / CLOCKS_PER_SEC);
	//Compare
	for(int i=0; i<n; i++){
		vector<KeyPoint> kpts;
		Mat dst;
		vector< vector<DMatch> > matches12, matches21;
        	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
		char s4[100] = "keypoints";
       		char s5[100] = "descriptors";
		char number[20];
		sprintf(number, "%d", i);
		strcat(s4,number);
		strcat(s5,number);
		
		//Read the file and Compare
		clock_t begin_read = clock();
		FileNode kptFileNode0 = fs2[s4];
  		//
  		read( kptFileNode0, kpts);
		//
		FileNode kptFileNode00 = fs2[s5];
        	read( kptFileNode00, dst );
		clock_t end_read = clock();
		Time_To_Read = Time_To_Read + (double(end_read - begin_read) / CLOCKS_PER_SEC);

        	matcher->knnMatch( descriptorsCompare, dst, matches12, 2 );
        	matcher->knnMatch( dst, descriptorsCompare, matches21, 2 );
		
		clock_t begin_compare = clock();
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
		clock_t end_compare = clock();
		Time_To_Compare = Time_To_Compare + (double(end_compare - begin_compare) / CLOCKS_PER_SEC);
		//cout<<endl<<i<<" : "<<better_matches.size()<<endl;
		if(i==n-2)
			i=n;
		if(better_matches.size() > tmp){
                        tmp = better_matches.size();
			best_matches = better_matches;
			Bestkpts = kpts; 
                        currentImg = i;
                }

	}
		clock_t begin_close = clock();
		fs2.release();
		clock_t end_close = clock();
		Time_To_Close_File = double(end_close - begin_close) / CLOCKS_PER_SEC;
		cout<<endl<<"Open File time : "<<Time_To_Open_File<<endl;
		cout<<"Close File time : "<<Time_To_Close_File<<endl;
		cout<<"Read Time : "<<Time_To_Read<<endl;
		cout<<"Averrage : "<<Time_To_Read/n<<endl;
		cout<<"Compare Time : "<<Time_To_Compare<<endl;
		cout<<"Average : "<<Time_To_Compare/n<<endl;
		cout<<"Perfect match frame : "<<currentImg<<endl;

	
	
		char s3[100] = "CompareRoom/";
		strcat(s3,s[currentImg]);
		cv::Mat BestImage;
		BestImage = cv::imread(s3,CV_LOAD_IMAGE_UNCHANGED);
		/*system("rm -rf ORBCompareKeyPoint.txt");
		FILE *fin2;
                fin2 = fopen("ORBCompareKeyPoint.txt","a");
		for(int i=0; i< 100;i++){
			fprintf(fin2,"%f\t%f\n",keypointsCompare[i].pt.x,keypointsCompare[i].pt.y);
		}
		fclose(fin2);
		*/
		//Dmatch Result saved
		FILE *fp2;
	        fp2 = fopen("OpencvORBMatches.txt","w+t");
		for(int i = 0; i<best_matches.size(); i++){
			int idx1=best_matches[i].trainIdx;
    			int idx2=best_matches[i].queryIdx;
			fprintf(fp2,"%d\t%f\t%f\t%f\t%f\n",i,keypointsCompare[idx1].pt.x,keypointsCompare[idx1].pt.y,Bestkpts[idx2].pt.x,Bestkpts[idx2].pt.y);		
		}	
		fclose(fp2);
		Mat output;
    		drawMatches(CompareImage, keypointsCompare, BestImage, Bestkpts, best_matches, output);
    		//imshow("Matches result",output);
    		waitKey(0);
		
		




}
