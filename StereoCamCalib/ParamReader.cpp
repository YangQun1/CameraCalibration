#include "ParamReader.h"

ParamReader::ParamReader(string calib_input_filename)
{
	FileStorage fs_input(calib_input_filename, FileStorage::READ);

	fs_input["Input_Type"] >> input_type;
	fs_input["Camera_Type"] >> camera_type;
	if (input_type == "ImageFolder"){
		fs_input["Input1"] >> img_path_left;
		if (camera_type == "Stereo"){
			fs_input["Input2"] >> img_path_right;
		}
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
}

ParamReader::~ParamReader()
{
}