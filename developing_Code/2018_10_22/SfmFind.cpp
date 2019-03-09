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

    if(argc != 10){
        cout << "usage:match <image1> <image2> <image3> <image4> <image5> <image6> <image7> <image8> <ratio>\n" ;
        exit(-1);
    }
	

	
	
	const int row = 300; //第一張為參考張，其他依序處理
    const int column = 5; // 0 是第幾張照片, 1 bettermatch的size, 2 Match 序號, 3 基準圖x座標,  4 基準圖y座標, 5 比較圖x座標, 6 比較圖y座標
    int arr2[row][column]; 
	int arr3[row][column];
	int arr4[row][column];
	int arr5[row][column];
	int arr6[row][column];
	int arr7[row][column];
	int arr8[row][column];
    	
	for(int k = 2; k <= 8 ;k++){
		double ratio = (double)atof(argv[9]);
		string image1_name=string(argv[1]);
		string image2_name=string(argv[k])
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
		if(k==2){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr2[i][0] = 2;
				arr2[i][1] = better_matches.size();
				arr2[i][2] = i;
				arr2[i][3] = keypoints1[idx1].pt.x;
				arr2[i][4] = keypoints1[idx1].pt.y;
				arr2[i][5] = keypoints2[idx1].pt.x;
				arr2[i][6] = keypoints2[idx1].pt.y;
				
				
			}
		}
		if(k==3){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr3[i][0] = 3;
				arr3[i][1] = better_matches.size();
				arr3[i][2] = i;
				arr3[i][3] = keypoints1[idx1].pt.x;
				arr3[i][4] = keypoints1[idx1].pt.y;
				arr3[i][5] = keypoints2[idx1].pt.x;
				arr3[i][6] = keypoints2[idx1].pt.y;
			}
		}
		if(k==4){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr4[i][0] = 4;
				arr4[i][1] = better_matches.size();
				arr4[i][2] = i;
				arr4[i][3] = keypoints1[idx1].pt.x;
				arr4[i][4] = keypoints1[idx1].pt.y;
				arr4[i][5] = keypoints2[idx1].pt.x;
				arr4[i][6] = keypoints2[idx1].pt.y;
			}
		}
		if(k==5){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr5[i][0] = 5;
				arr5[i][1] = better_matches.size();
				arr5[i][2] = i;
				arr5[i][3] = keypoints1[idx1].pt.x;
				arr5[i][4] = keypoints1[idx1].pt.y;
				arr5[i][5] = keypoints2[idx1].pt.x;
				arr5[i][6] = keypoints2[idx1].pt.y;
			}
		}
		if(k==6){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr6[i][0] = 6;
				arr6[i][1] = better_matches.size();
				arr6[i][2] = i;
				arr6[i][3] = keypoints1[idx1].pt.x;
				arr6[i][4] = keypoints1[idx1].pt.y;
				arr6[i][5] = keypoints2[idx1].pt.x;
				arr6[i][6] = keypoints2[idx1].pt.y;
			}
		}
		if(k==7){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr7[i][0] = 7;
				arr7[i][1] = better_matches.size();
				arr7[i][2] = i;
				arr7[i][3] = keypoints1[idx1].pt.x;
				arr7[i][4] = keypoints1[idx1].pt.y;
				arr7[i][5] = keypoints2[idx1].pt.x;
				arr7[i][6] = keypoints2[idx1].pt.y;
			}
		}
		if(k==8){
			for(int i = 0; i<better_matches.size();i++){
				int idx1=better_matches[i].trainIdx;
				int idx2=better_matches[i].queryIdx;
				arr8[i][0] = 8;
				arr8[i][1] = better_matches.size();
				arr8[i][2] = i;
				arr8[i][3] = keypoints1[idx1].pt.x;
				arr8[i][4] = keypoints1[idx1].pt.y;
				arr8[i][5] = keypoints2[idx1].pt.x;
				arr8[i][6] = keypoints2[idx1].pt.y;
			}
		}
	}
		//choose the biggest one
	cout<<endl<<" 2 ="<<arr2[1][1];
	cout<<endl<<" 3 ="<<arr3[1][1];
	cout<<endl<<" 4 ="<<arr4[1][1];
	cout<<endl<<" 5 ="<<arr5[1][1];
	cout<<endl<<" 6 ="<<arr6[1][1];
	cout<<endl<<" 7 ="<<arr7[1][1];
	cout<<endl<<" 8 ="<<arr8[1][1];
	
	   //the biggest one compare
	
		
}


    


