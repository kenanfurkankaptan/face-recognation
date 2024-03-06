#include "image_provider.h"

#include <filesystem>

namespace fs = std::filesystem;

image_provider::image_provider(source_type type, int api_preference) : type{type} {
	std::cout << "Initiating face detector\n";

	if (type == from_camera) {
		cap = cv::VideoCapture(api_preference);
	}
};

cv::Mat image_provider::get_image(std::string path) {
	cv::Mat img;
	if (type == from_camera && cap.isOpened()) {
		cap.read(img);
	} else if (type == from_file && !path.empty() && fs::exists(path)) {
		img = cv::imread(path);
	}

	return img;
}
