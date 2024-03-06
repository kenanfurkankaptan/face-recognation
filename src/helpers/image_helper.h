#ifndef IMAGE_HELPER_H_
#define IMAGE_HELPER_H_

#include <opencv2/opencv.hpp>

namespace image_helper {

	std::vector<cv::Mat> crop_resize_faces(cv::Mat image, std::vector<cv::Rect>);
	cv::Mat crop_resize_faces(cv::Mat image, cv::Rect pos);
	int save_image(std::string filename, cv::Mat image);
	cv::Mat label_faces(cv::Rect pos, std::string name, cv::Mat image);

};	// namespace image_helper

#endif /* IMAGE_HELPER_H_ */
