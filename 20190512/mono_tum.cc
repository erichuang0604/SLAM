/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <dirent.h> // for linux systems
#include <sys/stat.h> // for linux systems
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

    #include <opencv2/highgui/highgui.hpp>
    #include <string>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
int main(int argc, char **argv)
{
    FILE *fposition;
    fposition=fopen("CameraPosition.txt","w+t");
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }
    system("rm -rf /home/erichuang0604/ORB_SLAM2/rgbResult/*.txt");
    system("rm -rf /home/erichuang0604/ORB_SLAM2/rgbResult/Verify/*.jpg");
    system("rm -rf /home/erichuang0604/ORB_SLAM2/rgbResult/template.bin");
    system("rm -rf /home/erichuang0604/ORB_SLAM2/rgbResult/descriptor/*.txt");
    system("rm -rf /home/erichuang0604/ORB_SLAM2/rgbResult/keypoint/*.txt");
    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    //int nImages = vstrImageFilenames.size();
    //20190408
    int nImages = 1000000;
    // Open a video and get the initial image
    cv::VideoCapture video;
    //if (!video.open("data/KITTI_07_L/%06d.png")) return -1;
    if (!video.open("rgb/%d.jpg")) return -1;
    //if (!video.open("Output.mp4")) return -1;
    //20190408
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    int index;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        //im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
	//im = cv::imread(folder + filenames[ni]);
	cout<<"1"<<endl;
        video >> im;
	cout<<"2"<<endl;
        double tframe = vTimestamps[ni];
	cout<<"3"<<endl;
        if(im.empty())
        {
	SLAM.Shutdown();
	    fclose(fposition);
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;

            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
	cout<<"4"<<endl;
        // Pass the image to the SLAM system
	index = ni;
        SLAM.TrackMonocular(im,tframe,index);
	//20190505
	


        cout<<"5"<<endl;
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    // Save camera trajectory
float x,y,z;
		SLAM.SaveKeyFrameTrajectoryTUM2("KeyFrameTrajectory.txt",&x,&y,&z); //get x, y, z
fprintf(fposition,"%f\t%f\t%f\n",x,y,z);
cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<endl;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;


// Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
