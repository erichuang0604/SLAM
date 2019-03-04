    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <string>
    #include <iostream>

    using namespace cv;
    using namespace std;

int main(){
    vector<String> images; // notice here that we are using the Opencv’s embedded “String” class
    String patern = "rgb/"; // again we are using the Opencv’s embedded “String” class
    glob(patern, images);
    for(int i=0;i<images.size();i++){
    	Mat img = imread(images[i]);
	if(!img.data) 
		cout<<"Cant open"<<images[i]<<endl;
    	else
    		imshow("Image",img);
	waitKey(20);
    }
    return 0;
}
