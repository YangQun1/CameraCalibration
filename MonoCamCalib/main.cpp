#include <io.h>

#include <string>
#include <vector>
#include <iostream>
#include <time.h>

#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;


void getFiles(string path, vector<string>& files);

#define CALIBRATE	

int main() 
{
	// 获取输入参数
	string img_path;
	string input_type;
	string camera_type;
	int radial_dist_num;
	int zero_tangential_dist;
	string output_filename;
	int iter_subpixel;
	double eps_subpixel;
	int iter_calib;
	double eps_calib;
	Size board_size;
	Size chess_size;

	string calib_input_filename = "./calibration_input.yml";
	// string calib_input_filename = "./calibration_results.yml";

	FileStorage fs_input(calib_input_filename, FileStorage::READ);
	
	fs_input["Input_Type"] >> input_type;
	if (input_type == "ImageFolder"){
		fs_input["Input1"] >> img_path;
	}

	fs_input["BoardSize_Width"] >> board_size.width;
	fs_input["BoardSize_Height"] >> board_size.height;

	fs_input["ChessSize_Width"] >> chess_size.width;
	fs_input["ChessSize_Height"] >> chess_size.height;

	fs_input["Radial_Dist_Num"] >> radial_dist_num;
	fs_input["Zero_Tangential_Dist"] >> zero_tangential_dist;

	fs_input["Output_File_Name"] >> output_filename;

	fs_input["Iter_Subpixel"] >> iter_subpixel;
	fs_input["EPS_Subpixel"] >> eps_subpixel;

	fs_input["Iter_Calibration"] >> iter_calib;
	fs_input["EPS_Calibration"] >> eps_calib;

	fs_input.release();

	// 读取输入数据
	// string img_path = "E:/VSProject/CameraCalibration/DataSet/slr/";
	vector<string> img_files;
	getFiles(img_path, img_files);
	int img_count = img_files.size();

#ifdef CALIBRATE	// 重新标定
	// 提取棋盘角点
	Size img_size;
	int valid_img_count = 0;
	vector<Point2f> corners;
	vector<vector<Point2f>> all_corners;
	namedWindow("corners", 0);
	cvResizeWindow("corners", 1072, 712);
	for (int i = 0; i < img_count; i++){

		Mat img = imread(img_path + img_files[i]);

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
		bool found = findChessboardCorners(img_grey, board_size, corners, CALIB_CB_ADAPTIVE_THRESH \
			+ CALIB_CB_NORMALIZE_IMAGE \
			+ CALIB_CB_FAST_CHECK);

		if (!found) continue;

		valid_img_count++;

		// 精细化到亚像素
		// 注意：亚像素精细化过程非常重要，影响着标定的精度
		cornerSubPix(img_grey, corners, Size(11, 11), Size(-1, -1),\
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, iter_subpixel, eps_subpixel));
		all_corners.push_back(corners);
		 
		// 绘制角点并显示
		drawChessboardCorners(img, board_size, corners, true);
		imshow("corners", img);
		waitKey(0);
	}
	cv::destroyWindow("corners");

	// 构造三维的目标点，在每个视角下，使一个角点对应一个目标点
	vector<vector<Point3f>> all_object_points;
	// Size chess_size = { 25, 25 };
	for (int i = 0; i < valid_img_count; i++){
		vector<Point3f> object_points;
		for (int h = 0; h < board_size.height; h++){
			for (int w = 0; w < board_size.width; w++){
				Point3f object_point;
				object_point.x = w*chess_size.width;
				object_point.y = h*chess_size.height;
				object_point.z = 0;
				object_points.push_back(object_point);
			}
		}
		all_object_points.push_back(object_points);
	}

	// 开始标定
	Mat intrinsic_matrix = Mat::zeros(Size(3, 3), CV_32FC1);
	Mat dist_coeffs = Mat::zeros(Size(1, 8), CV_32FC1);
	vector<Mat> rvecs, tvecs;

	//<--	此处，使用不同阶数的畸变项效果相差很大
	//		因为在使用非线性优化对畸变系数进行求解的过程中，仅仅是最小化棋盘角点的重投影误差，
	//		如果选取的畸变阶数不合适，可能会造成棋盘角点重投影误差很小，但图像边缘（没有棋盘角点的位置等）
	//		重投影误差很大的现象，类似于过拟合问题。
	//-->
	int flag = 0;
	if (radial_dist_num == 2){
		flag = flag | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (radial_dist_num == 3){
		flag = flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	else if (radial_dist_num == 4){
		flag = flag | CV_CALIB_RATIONAL_MODEL;
		flag = flag | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
	}
	if (zero_tangential_dist){
		flag = flag | CV_CALIB_ZERO_TANGENT_DIST;
	}

	double rms = calibrateCamera(all_object_points, all_corners, img_size, \
		intrinsic_matrix, dist_coeffs, rvecs, tvecs, flag,  \
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, iter_calib, eps_calib));

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
	double reproj_error, total_error = 0;
	for (int i = 0; i < valid_img_count; i++){
		// 计算重投影点
		projectPoints(all_object_points[i], rvecs[i], tvecs[i], \
			intrinsic_matrix, dist_coeffs, reproj_points);

		// 计算重投影误差
		reproj_error = 0;
		int points_num = reproj_points.size();
		for (int j = 0; j < points_num; j++){
			reproj_error += pow(reproj_points[j].x - all_corners[i][j].x, 2.0)+ \
				pow(reproj_points[j].y - all_corners[i][j].y, 2.0);
		}
		reproj_error = reproj_error / points_num;

		total_error += reproj_error;
	}
	total_error = total_error / img_count;
	cout << "average re-projection error: " << total_error << endl;

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

	fs_out << "Camera_Matrix" << intrinsic_matrix;

	cvWriteComment(*fs_out, "perspective camera distortion module:k1, k2, p1, p2[, k3[, k4, k5, k6]]", 0);
	fs_out << "Distortion_Coefficients" << dist_coeffs;

	fs_out << "Avg_Reprojection_Error" << total_error;

	fs_out.release();

#else	// 从文件中读取已经标定得到的参数
	string calib_results_file = "./calibration_results_matlab.yml";
	FileStorage fs(calib_results_file, FileStorage::READ);

	Mat intrinsic_matrix;
	Mat dist_coeffs;

	fs["Camera_Matrix"] >> intrinsic_matrix;
	fs["Distortion_Coefficients"] >> dist_coeffs;

	fs.release();
#endif 

	// 去畸变实验
	namedWindow("undistort", 0);
	cvResizeWindow("undistort", 1072, 712);
	for (int i = 0; i < img_count; i++){
		Mat img = imread(img_path + img_files[i]);
		Mat img_undist;
		undistort(img, img_undist, intrinsic_matrix, dist_coeffs);
		imshow("undistort", img_undist);
		waitKey(0);
	}
	cv::destroyWindow("undistort");
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
