#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <Windows.h>
#include <io.h>
#include <direct.h>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\video.hpp>

using namespace std;
using namespace cv;

int main()
{
	// 控制参数
	bool isStereo = 0;
	int frame_num = 30;
	int camID1 = 0 , camID2 = 1;
	int captureInterval = 2;	// second
	string savePath = "E:/VSProject/CameraCalibration/DataSet/captured/";

	// 做准备工作
	bool isOpened1, isOpened2;
	string imgName1, imgName2;
	string savePath1, savePath2;
	VideoCapture inputCapture1, inputCapture2;

	if (isStereo){
		savePath1 = savePath + "left/";
		savePath2 = savePath + "right/ ";
		imgName1 = "left";
		imgName2 = "right";
		inputCapture1.open(camID1);
		inputCapture2.open(camID2);
		isOpened1 = inputCapture1.isOpened();
		isOpened2 = inputCapture2.isOpened();
		if (!isOpened1 | !isOpened2){
			cerr << "open camera failed" << endl;
			return -1;
		}
	}
	else{
		savePath1 = savePath;
		imgName1 = "image";
		inputCapture1.open(camID1);
		isOpened1 = inputCapture1.isOpened();
		if (!isOpened1){
			cerr << "open camera failed" << endl;
			return -1;
		}
	}

	// 检查保存路径是否存在，若不存在，则创建
	if (-1 == _access(savePath.c_str(), 0)){
		_mkdir(savePath.c_str());
	}
	if (isStereo){
		if (-1 == _access(savePath1.c_str(), 0)){
			_mkdir(savePath1.c_str());
		}
		if (-1 == _access(savePath2.c_str(), 0)){
			_mkdir(savePath2.c_str());
		}
	}
	// 定时器，用于定时的拍照
	LARGE_INTEGER li;
	LONGLONG last, now, freq;

	QueryPerformanceFrequency(&li);
	freq = li.QuadPart;

	QueryPerformanceCounter(&li);
	now = last = li.QuadPart;

	LONGLONG countInterval = captureInterval*freq;

	// 开始采集图像
	stringstream frame_num_ss;
	string frame_num_s;
	frame_num_ss << setw(3) << setfill('0') << frame_num;
	frame_num_ss >> frame_num_s;

	for (int i = 0; i < frame_num;){
		
		stringstream inter;
		string index;
		inter << setw(3) << setfill('0') << i;
		inter >> index;

		string filename1 = savePath1 + imgName1 + index + ".png";
		string filename2;
		if (isStereo){
			filename2 = savePath2 + imgName2 + index + ".png";
		}

		// 获取
		Mat image1, image2;
		inputCapture1 >> image1;
		if (isStereo){
			inputCapture2 >> image2;
		}

		// 实时显示
		Mat temp1 = image1.clone();
		putText(temp1, index + "/" + frame_num_s, Point(50, 60), \
			FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);	// 显示已采集的帧数
		imshow("Image1", temp1);
		if (isStereo){
			Mat temp2 = image2.clone();
			putText(temp2, index + "/" + frame_num_s, Point(50, 60), \
				FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);	// 显示已采集的帧数
			imshow("Image2", temp2);
		}
		waitKey(20);

		// 定时保存
		QueryPerformanceCounter(&li);
		now = li.QuadPart;
		if ((now - last) > countInterval){
			imwrite(filename1, image1);
			if (isStereo){
				imwrite(filename2, image2);
			}
			i++;
			last = now;
		}
	}

	// 关闭摄像头
	inputCapture1.release();
	if (isStereo){
		inputCapture2.release();
	}

	cout << "Capture Image Finished" << endl << "You Can Start to Calibrate Your Camera" << endl;
	return 0;
}