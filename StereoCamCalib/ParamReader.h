#ifndef PARAM_READER_H
#define PARAM_READER_H

#include <opencv2\core\core.hpp>
#include <string>

using namespace std;
using namespace cv;

class ParamReader
{
public:
	ParamReader(string calib_input_filename);
	~ParamReader();

public:
	string img_path_left;
	string img_path_right;
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


};
#endif