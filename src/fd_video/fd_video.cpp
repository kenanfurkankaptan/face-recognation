#include "fd_video.h"

#include <filesystem>

namespace fs = std::filesystem;

int fd_video::load_model(std::string path) {
	if (path == "" || path.empty()) {
		std::cout << "fd_video::load_model: path is empty" << std::endl;
		return 1;
	}

	if (!fs::exists(path)) {
		std::cout << "fd_video::load_model: path is not exist -- " << path << std::endl;
		return 1;
	}

	// if (this->face_cascade.empty()) {
	if (!face_cascade.load(path)) {
		std::cerr << "fd_video::load_model: cascade file is empty\n";
		return 1;
	}
}

int fd_video::select_source(std::string path) {
	cap = cv::VideoCapture(0);
}

std::vector<cv::Mat> fd_video::get_faces(std::string path) {
	if (this->face_cascade.empty()) {
		std::cerr << "fd_video::load_model: cascade model does not exist\n";
		return {};
	}

	std::vector<cv::Mat> face_images;

	if (path == "" || path.empty()) {
		cap.read(img);
	} else {
		img = cv::imread(path);
	}

	cv::Mat img_grey;

	/** preprocess */
	cvtColor(img, img_grey, cv::COLOR_BGRA2GRAY);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(img_grey, faces, 1.1, 10);

	for (auto i : faces) {
		face_images.push_back(img_grey(i));
	}

	for (int i = 0; i < faces.size(); i++) {
		rectangle(img, faces[i].tl(), faces[i].br(), cv::Scalar(255, 0, 255), 3);
	}
	imshow("Image", img);

	return face_images;
}