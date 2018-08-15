#include <io.h>

#include <string>
#include <vector>
#include <iostream>
#include <time.h>

#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "ParamReader.h"

using namespace cv;
using namespace std;


void getFiles(string path, vector<string>& files);

double calibSingleCam(
	vector<string>& img_files,
	vector<string>& img_input_valid,
	ParamReader& param_reader,
	Mat& intrinsic_matrix, Mat& dist_coeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs);

double calibStereoCam(
	vector<string>& img_files_left, vector<string>& img_files_right,
	vector<string>& img_input_valid,
	ParamReader& param_reader,
	Mat& intrinsic_matrix_left, Mat& dist_coeffs_left,
	Mat& intrinsic_matrix_right, Mat& dist_coeffs_right,
	Mat& R, Mat& t,
	Size& img_size);


// #define CALIBRATE	

int main() 
{
	// 获取输入参数
	string calib_input_filename = "./calibration_input.yml";
	ParamReader param_reader(calib_input_filename);
	

	// 读取输入数据
	vector<string> img_files_left, img_files_right;
	getFiles(param_reader.img_path_left, img_files_left);
	getFiles(param_reader.img_path_right, img_files_right);


#ifdef CALIBRATE	// 重新标定
	// 标定相机
	Size img_size;
	vector<string> img_input_valid;
	Mat intrinsic_matrix_left, dist_coeffs_left;
	Mat intrinsic_matrix_right, dist_coeffs_right;
	Mat R, t;
	double rms = calibStereoCam(img_files_left, img_files_right,
		img_input_valid,
		param_reader,
		intrinsic_matrix_left, dist_coeffs_left,
		intrinsic_matrix_right, dist_coeffs_right,
		R, t, img_size);

	if (rms < 0){
		cerr << "calibrate error" << endl;
		return -1;
	}

	// 保存标定结果
	string out_file_name = "./calibration_results.yml";
	FileStorage fs_out(out_file_name, FileStorage::WRITE);

	time_t tm;
	struct tm t2;
	time(&tm);
	localtime_s(&t2, &tm);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", &t2);
	fs_out << "Calibration_Time" << buf;

	fs_out << "Image_Size" << img_size;

	fs_out << "Camera_Matrix_Left" << intrinsic_matrix_left;

	cvWriteComment(*fs_out, "perspective camera distortion module:k1, k2, p1, p2[, k3[, k4, k5, k6]]", 0);
	fs_out << "Distortion_Coefficients_Left" << dist_coeffs_left;

	fs_out << "Camera_Matrix_Right" << intrinsic_matrix_right;

	cvWriteComment(*fs_out, "perspective camera distortion module:k1, k2, p1, p2[, k3[, k4, k5, k6]]", 0);
	fs_out << "Distortion_Coefficients_Right" << dist_coeffs_right;

	fs_out << "Rotation_Matrix" << R;
	fs_out << "Translation_Vector" << t;

	fs_out << "Avg_Reprojection_Error" << rms;

	fs_out.release();

#else	// 从文件中读取已经标定得到的参数
	string calib_results_file = "./calibration_results.yml";
	FileStorage fs(calib_results_file, FileStorage::READ);

	Mat intrinsic_matrix_left, intrinsic_matrix_right;
	Mat dist_coeffs_left, dist_coeffs_right;
	Mat R, t;
	Size img_size;

	fs["Camera_Matrix_Left"] >> intrinsic_matrix_left;
	fs["Distortion_Coefficients_Left"] >> dist_coeffs_left;
	fs["Camera_Matrix_Right"] >> intrinsic_matrix_right;
	fs["Distortion_Coefficients_Right"] >> dist_coeffs_right;
	fs["Rotation_Matrix"] >> R;
	fs["Translation_Vector"] >> t;
	fs["Image_Size"] >> img_size;

	fs.release();
#endif 

	// 立体校正实验
	// 立体匹配与深度恢复实验
	StereoSGBM sgbmMatcher;

	Mat R1, R2, P1, P2, Q;
	stereoRectify(intrinsic_matrix_left, dist_coeffs_left,
		intrinsic_matrix_right, dist_coeffs_right, img_size, R, t,
		R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, 0);

	Mat map1_left, map2_left;
	Mat map1_right, map2_right;
	initUndistortRectifyMap(intrinsic_matrix_left, dist_coeffs_left,
		R1, P1, img_size, CV_32FC1, map1_left, map2_left);
	initUndistortRectifyMap(intrinsic_matrix_right, dist_coeffs_right,
		R2, P2, img_size, CV_32FC1, map1_right, map2_right);

	namedWindow("rectify", 0);
	cvResizeWindow("rectify", 1800, 600);
	namedWindow("depth", 0);
	cvResizeWindow("depth", 900, 600);
	int img_count = img_files_left.size();
	for (int i = 0; i < img_count; i++){
		Mat img_left = imread(param_reader.img_path_left + img_files_left[i]);
		Mat img_right = imread(param_reader.img_path_right + img_files_right[i]);

		Mat img_rectified_left, img_rectified_right;
		remap(img_left, img_rectified_left, map1_left, map2_left, CV_INTER_LINEAR);
		remap(img_right, img_rectified_right, map1_right, map2_right, CV_INTER_LINEAR);

		// 画出对极线，以备显示
		Mat img_rectify = Mat::zeros(img_size.height, img_size.width * 2, img_left.type());
		img_rectified_left.copyTo(img_rectify(Rect(0, 0, img_size.width, img_size.height)));
		img_rectified_right.copyTo(img_rectify(Rect(img_size.width, 0, img_size.width, img_size.height)));
		for (int j = 0; j < img_rectify.rows; j += 48)
			line(img_rectify, Point(0, j), Point(img_rectify.cols, j), Scalar(0, 255, 0), 1, 8);

		// 转为灰度
		Mat img_rectified_right_grey, img_rectified_left_grey;
		if (img_rectified_right.channels() == 3){
			cvtColor(img_rectified_right, img_rectified_right_grey, CV_RGB2GRAY);
			cvtColor(img_rectified_left, img_rectified_left_grey, CV_RGB2GRAY);
		}
		else{
			img_rectified_right_grey = img_rectified_right;
			img_rectified_left_grey = img_rectified_left;
		}

		// 进行双目匹配
		int cn = img_rectified_right_grey.channels();
		int SADWindowSize = 21;
		sgbmMatcher.preFilterCap = 63;
		sgbmMatcher.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
		sgbmMatcher.P1 = 8 * cn*sgbmMatcher.SADWindowSize*sgbmMatcher.SADWindowSize;
		sgbmMatcher.P2 = 32 * cn*sgbmMatcher.SADWindowSize*sgbmMatcher.SADWindowSize;
		sgbmMatcher.minDisparity = 0;
		sgbmMatcher.numberOfDisparities = ((img_rectified_right_grey.cols / 8) + 15) & -16;	//32;;
		sgbmMatcher.uniquenessRatio = 10;
		sgbmMatcher.speckleWindowSize = 100;
		sgbmMatcher.speckleRange = 32;
		sgbmMatcher.disp12MaxDiff = 1;
		sgbmMatcher.fullDP = false;

		Mat disparity;
		sgbmMatcher(img_rectified_left_grey, img_rectified_right_grey, disparity);

		// 恢复深度
		Mat _3DPoint, depthImage;
		vector<Mat> channels;
		reprojectImageTo3D(disparity, _3DPoint, Q);
		split(_3DPoint, channels);
		channels[2].convertTo(depthImage, CV_8UC1);

		// 显示
		imshow("rectify", img_rectify);
		imshow("depth", depthImage);
		waitKey(0);
	}
	cv::destroyAllWindows();



 	return 0;
}

// 将目录中的所有文件加入向量列表
// 将目录中的所有文件名加入到files向量中
void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if (path[path.length() - 1] != '\\' && path[path.length() - 1] != '/'){
		path = path + '/';
	}
	if ((hFile = _findfirst(p.assign(path).append("*").c_str(), &fileinfo)) != -1){
		do{ //如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR)){
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append(fileinfo.name), files);
			}
			else
			{
				//files.push_back(p.assign(path).append(fileinfo.name));
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


// 标定单个相机
double calibSingleCam(
	vector<string>& img_files, 
	vector<string>& img_input_valid,
	ParamReader& param_reader, 
	Mat& intrinsic_matrix, Mat& dist_coeffs, 
	vector<Mat>& rvecs, vector<Mat>& tvecs)
{
	Size img_size;
	vector<Point2f> corners;
	vector<vector<Point2f>> all_corners;

	int img_count = img_files.size();

	namedWindow("corners", 0);
	cvResizeWindow("corners", 1072, 712);
	for (int i = 0; i < img_count; i++){

		string img_filename = param_reader.img_path_left + img_files[i];
		Mat img = imread(img_filename);

		// 获取图像的size
		if (i == 0){
			img_size.width = img.cols;
			img_size.height = img.rows;
		}

		// 转成灰度图
		Mat img_grey;
		if (img.channels() == 3 || img.channels() == 4){
			cvtColor(img, img_grey, CV_RGB2GRAY);
		}
		else{
			img_grey = img;
		}

		// 提取角点
		bool found = findChessboardCorners(img_grey, param_reader.board_size, corners, CALIB_CB_ADAPTIVE_THRESH \
			+ CALIB_CB_NORMALIZE_IMAGE \
			+ CALIB_CB_FAST_CHECK);

		if (!found) continue;

		img_input_valid.push_back(img_filename);

		// 精细化到亚像素
		// 注意：亚像素精细化过程非常重要，影响着标定的精度
		cornerSubPix(img_grey, corners, Size(11, 11), Size(-1, -1), \
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, param_reader.iter_subpixel, param_reader.eps_subpixel));
		all_corners.push_back(corners);

		// 绘制角点并显示
		drawChessboardCorners(img, param_reader.board_size, corners, true);
		imshow("corners", img);
		waitKey(200);
	}
	cv::destroyWindow("corners");

	// 构造三维的目标点，在每个视角下，使一个角点对应一个目标点
	vector<vector<Point3f>> all_object_points;
	int valid_img_count = img_input_valid.size();
	for (int i = 0; i < valid_img_count; i++){
		vector<Point3f> object_points;
		for (int h = 0; h < param_reader.board_size.height; h++){
			for (int w = 0; w < param_reader.board_size.width; w++){
				Point3f object_point;
				object_point.x = w*param_reader.chess_size.width;
				object_point.y = h*param_reader.chess_size.height;
				object_point.z = 0;
				object_points.push_back(object_point);
			}
		}
		all_object_points.push_back(object_points);
	}

	// 开始标定
	intrinsic_matrix = Mat::zeros(Size(3, 3), CV_32FC1);
	dist_coeffs = Mat::zeros(Size(1, 8), CV_32FC1);

	//<--	此处，使用不同阶数的畸变项效果相差很大
	//		因为在使用非线性优化对畸变系数进行求解的过程中，仅仅是最小化棋盘角点的重投影误差，
	//		如果选取的畸变阶数不合适，可能会造成棋盘角点重投影误差很小，但图像边缘（没有棋盘角点的位置等）
	//		重投影误差很大的现象，类似于过拟合问题。
	//-->
	int flag = 0;
	if (param_reader.radial_dist_num == 2){
		flag = flag | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (param_reader.radial_dist_num == 3){
		flag = flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (param_reader.radial_dist_num == 4){
		flag = flag | CV_CALIB_RATIONAL_MODEL;
		flag = flag | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	if (param_reader.zero_tangential_dist){
		flag = flag | CV_CALIB_ZERO_TANGENT_DIST;
	}

	double rms = calibrateCamera(all_object_points, all_corners, img_size, \
		intrinsic_matrix, dist_coeffs, rvecs, tvecs, flag, \
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, param_reader.iter_calib, param_reader.eps_calib));

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;
	bool ok = checkRange(intrinsic_matrix) && checkRange(dist_coeffs);
	if (ok){
		cout << "Intrinsic-Matrix:" << endl << intrinsic_matrix << endl;
		cout << "Dist-Coeffs:" << endl << dist_coeffs << endl;
	}
	else{
		cout << "Bad estimated Intrinsic-Matrix or Dist-Coffes" << endl;
	}

	// 评价标定结果  
	vector<Point2f> reproj_points;
	double reproj_error;
	double average_reproj_error = 0;
	for (int i = 0; i < valid_img_count; i++){
		// 计算重投影点
		projectPoints(all_object_points[i], rvecs[i], tvecs[i], \
			intrinsic_matrix, dist_coeffs, reproj_points);

		// 计算重投影误差
		reproj_error = 0;
		int points_num = reproj_points.size();
		for (int j = 0; j < points_num; j++){
			reproj_error += pow(reproj_points[j].x - all_corners[i][j].x, 2.0) + \
				pow(reproj_points[j].y - all_corners[i][j].y, 2.0);
		}
		reproj_error = reproj_error / points_num;

		average_reproj_error += reproj_error;
	}
	average_reproj_error = average_reproj_error / img_count;
	cout << "average re-projection error: " << average_reproj_error << endl;

	return average_reproj_error;
}


double calibStereoCam(
	vector<string>& img_files_left, vector<string>& img_files_right,
	vector<string>& img_input_valid,
	ParamReader& param_reader,
	Mat& intrinsic_matrix_left, Mat& dist_coeffs_left,
	Mat& intrinsic_matrix_right, Mat& dist_coeffs_right,
	Mat& R, Mat& t,
	Size& img_size)
{
	vector<vector<Point2f>> all_corners_left;
	vector<vector<Point2f>> all_corners_right;

	int img_count_left = img_files_left.size();
	int img_count_right = img_files_right.size();
	if (img_count_left != img_count_right)
	{
		cerr << "left image and right image have to be in same number" << endl;
		return -1.0;
	}

	// 提取左图像角点
	namedWindow("corners", 0);
	cvResizeWindow("corners", 1800, 600);
	for (int i = 0; i < img_count_left; i++){

		vector<Point2f> corners_left;
		vector<Point2f> corners_right;

		string img_left_filename = param_reader.img_path_left + img_files_left[i];
		string img_right_filename = param_reader.img_path_right + img_files_right[i];

		Mat img_left = imread(img_left_filename);
		Mat img_right = imread(img_right_filename);

		// 获取图像的size
		if (i == 0){
			img_size.width = img_left.cols;
			img_size.height = img_left.rows;
		}

		// 转成灰度图
		Mat img_grey_left, img_grey_right;
		if (img_left.channels() == 3 || img_left.channels() == 4){
			cvtColor(img_left, img_grey_left, CV_RGB2GRAY);
			cvtColor(img_right, img_grey_right, CV_RGB2GRAY);
		}
		else{
			img_grey_left = img_left;
			img_grey_right = img_right;
		}

		// 提取角点
		bool found_left = findChessboardCorners(img_grey_left, param_reader.board_size, corners_left, CALIB_CB_ADAPTIVE_THRESH \
			+ CALIB_CB_NORMALIZE_IMAGE \
			+ CALIB_CB_FAST_CHECK);
		bool found_right = findChessboardCorners(img_grey_right, param_reader.board_size, corners_right, CALIB_CB_ADAPTIVE_THRESH \
			+ CALIB_CB_NORMALIZE_IMAGE \
			+ CALIB_CB_FAST_CHECK);

		// 如果提取角点失败，放弃该图像对
		if (!found_left | !found_right){
			continue;
		}

		img_input_valid.push_back(img_left_filename);
		img_input_valid.push_back(img_right_filename);

		// 精细化到亚像素
		// 注意：亚像素精细化过程非常重要，影响着标定的精度
		cornerSubPix(img_grey_left, corners_left, Size(11, 11), Size(-1, -1), \
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, param_reader.iter_subpixel, param_reader.eps_subpixel));
		all_corners_left.push_back(corners_left);
		cornerSubPix(img_grey_right, corners_right, Size(11, 11), Size(-1, -1), \
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, param_reader.iter_subpixel, param_reader.eps_subpixel));
		all_corners_right.push_back(corners_right);

		// 绘制角点并显示
		drawChessboardCorners(img_left, param_reader.board_size, corners_left, true);
		drawChessboardCorners(img_right, param_reader.board_size, corners_right, true);
		Mat corner_img_pair = Mat::zeros(img_size.height, img_size.width * 2, img_left.type());
		img_left.copyTo(corner_img_pair(Rect(0, 0, img_size.width, img_size.height)));
		img_right.copyTo(corner_img_pair(Rect(img_size.width, 0, img_size.width, img_size.height)));
		imshow("corners", corner_img_pair);
		waitKey(200);
	}
	cv::destroyWindow("corners");

	// 构造三维的目标点，在每个视角下，使一个角点对应一个目标点
	vector<vector<Point3f>> all_object_points;
	int valid_img_count = img_input_valid.size()/2;
	for (int i = 0; i < valid_img_count; i++){
		vector<Point3f> object_points;
		for (int h = 0; h < param_reader.board_size.height; h++){
			for (int w = 0; w < param_reader.board_size.width; w++){
				Point3f object_point;
				object_point.x = w*param_reader.chess_size.width;
				object_point.y = h*param_reader.chess_size.height;
				object_point.z = 0;
				object_points.push_back(object_point);
			}
		}
		all_object_points.push_back(object_points);
	}

	// 开始标定
	intrinsic_matrix_left = Mat::zeros(Size(3, 3), CV_32FC1);
	dist_coeffs_left = Mat::zeros(Size(1, 8), CV_32FC1);
	intrinsic_matrix_right = Mat::zeros(Size(3, 3), CV_32FC1);
	dist_coeffs_right = Mat::zeros(Size(1, 8), CV_32FC1);

	//<--	此处，使用不同阶数的畸变项效果相差很大
	//		因为在使用非线性优化对畸变系数进行求解的过程中，仅仅是最小化棋盘角点的重投影误差，
	//		如果选取的畸变阶数不合适，可能会造成棋盘角点重投影误差很小，但图像边缘（没有棋盘角点的位置等）
	//		重投影误差很大的现象，类似于过拟合问题。
	//-->
	int flag = 0;
	if (param_reader.radial_dist_num == 2){
		flag = flag | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (param_reader.radial_dist_num == 3){
		flag = flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (param_reader.radial_dist_num == 4){
		flag = flag | CV_CALIB_RATIONAL_MODEL;
		flag = flag | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	if (param_reader.zero_tangential_dist){
		flag = flag | CV_CALIB_ZERO_TANGENT_DIST;
	}

	Mat E, F;
	double rms = stereoCalibrate(all_object_points, all_corners_left, all_corners_right, \
		intrinsic_matrix_left, dist_coeffs_left, intrinsic_matrix_right, dist_coeffs_right, \
		img_size, R, t, E, F, \
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, param_reader.iter_calib, param_reader.eps_calib), \
		flag);

	
	cout << "Re-projection error reported by stereoCalibrate: " << rms << endl;
	bool ok = checkRange(intrinsic_matrix_left) && checkRange(dist_coeffs_left) && \
		checkRange(intrinsic_matrix_right) && checkRange(dist_coeffs_right);
	if (ok){
		cout << "Left Intrinsic-Matrix:" << endl << intrinsic_matrix_left << endl;
		cout << "Left Dist-Coeffs:" << endl << dist_coeffs_left << endl;
		cout << "Right Intrinsic-Matrix:" << endl << intrinsic_matrix_right << endl;
		cout << "Right Dist-Coeffs:" << endl << dist_coeffs_right << endl;
	}
	else{
		cout << "Bad estimated Intrinsic-Matrix or Dist-Coffes" << endl;
	}

	
	cout << "average re-projection error: " << rms << endl;

	return rms;
}


