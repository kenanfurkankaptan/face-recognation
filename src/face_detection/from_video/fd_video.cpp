#include "fd_video.h"

#include <filesystem>

namespace fs = std::filesystem;

std::vector<cv::Mat> fd_video::get_faces(std::string path) {
	if (this->face_cascade.empty()) {
		std::cerr << "fd_video::load_model: cascade model does not exist\n";
		return {};
	}

	std::vector<cv::Mat> face_images;

	if (!cap.isOpened()) {
		std::cerr << "fd_video::get_faces: cap is not valid\n";
	}

	cap.read(img);

	cv::Mat img_grey;

	/** preprocess */
	cvtColor(img, img_grey, cv::COLOR_BGRA2GRAY);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(img_grey, faces, 1.1, 10);

	for (auto i : faces) {
		face_images.push_back(img_grey(i).clone());
	}

	for (int i = 0; i < faces.size(); i++) {
		rectangle(img, faces[i].tl(), faces[i].br(), cv::Scalar(255, 0, 255), 3);
	}

	imshow("Image", img);

	return face_images;
}