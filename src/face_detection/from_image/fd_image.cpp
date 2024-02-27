#include "fd_image.h"

#include <filesystem>

namespace fs = std::filesystem;

std::vector<cv::Mat> fd_image::get_faces(std::string path) {
	if (this->face_cascade.empty()) {
		std::cerr << "fd_image::load_model: cascade model does not exist\n";
		return {};
	}

	std::vector<cv::Mat> face_images;

	if (path == "" || path.empty()) {
		std::cerr << "fd_image::get_faces: path is empty\n";
	}

	img = cv::imread(path);

	cv::Mat img_grey;

	/** preprocess */
	cvtColor(img, img_grey, cv::COLOR_BGRA2GRAY);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(img_grey, faces, 1.1, 10);

	for (auto i : faces) {
		face_images.push_back(img_grey(i));
	}

	return face_images;
}