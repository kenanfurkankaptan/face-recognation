#ifndef IMAGE_HELPER_H_
#define IMAGE_HELPER_H_

#include <opencv2/opencv.hpp>

#include "../schemes/bgs.h"

namespace image_helper {

	std::vector<cv::Mat> crop_resize_faces(cv::Mat image, std::vector<cv::RotatedRect>);
	cv::Mat crop_resize_faces(cv::Mat image, cv::RotatedRect pos);
	int save_image(std::string filename, cv::Mat image);
	cv::Mat label_faces(cv::RotatedRect pos, std::string name, cv::Mat image);

	bgs calculate_bgs(cv::Mat& image);
	cv::Mat preprocess_image(cv::Mat& image, bgs values, bool gray_image = false);

};	// namespace image_helper

#endif /* IMAGE_HELPER_H_ */
