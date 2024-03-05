#include "face_detection.h"

#include <filesystem>

namespace fs = std::filesystem;

face_detection::face_detection(std::string model_path, source_type type) : type{type} {
	if (model_path == "" || model_path.empty()) {
		std::cout << "face_detection::load_model: path is empty" << std::endl;
	}

	if (!fs::exists(model_path)) {
		std::cout << "face_detection::load_model: path is not exist -- " << model_path << std::endl;
	}
	if (!face_cascade.load(model_path)) {
		std::cerr << "Error: Could not load Haar Cascade Classifier from " << model_path << std::endl;
	}

	if (type == from_camera) {
		cap = cv::VideoCapture(0);
	}
};

cv::Mat face_detection::get_image(std::string path) {
	cv::Mat img;
	if (type == from_camera && cap.isOpened()) {
		cap.read(img);
	} else if (type == from_file && !path.empty() && fs::exists(path)) {
		img = cv::imread(path);
	}

	return img;
}

std::vector<cv::Rect> face_detection::get_faces(cv::Mat img) {
	if (this->face_cascade.empty()) {
		std::cerr << "face_detection::load_model: cascade model does not exist\n";
		return {};
	}

	std::vector<cv::Mat> face_images;
	cv::Mat img_grey;

	/** preprocess */
	cvtColor(img, img_grey, cv::COLOR_BGRA2GRAY);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(img_grey, faces, 1.1, 10);

	return faces;
}

std::vector<cv::Mat> face_detection::crop_resize_faces(cv::Mat image, std::vector<cv::Rect> pos) {
	std::vector<cv::Mat> faces;
	cv::Mat img_grey;
	cvtColor(image, img_grey, cv::COLOR_BGRA2GRAY);

	for (auto p : pos) {
		cv::Mat temp;
		cv::resize(img_grey(p), temp, cv::Size(200, 200));

		faces.push_back(temp);
	}

	return faces;
}

cv::Mat face_detection::crop_resize_faces(cv::Mat image, cv::Rect pos) {
	cv::Mat img_grey, temp;
	cvtColor(image, img_grey, cv::COLOR_BGRA2GRAY);
	cv::resize(img_grey(pos), temp, cv::Size(200, 200));

	return temp;
}

cv::Mat face_detection::label_faces(cv::Rect pos, std::string name, cv::Mat image) {
	rectangle(image, pos.tl(), pos.br(), cv::Scalar(255, 0, 255), 3);
	putText(image, name, pos.tl(), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 69, 255), 3);

	return image;
}

int face_detection::save_image(std::string filename, cv::Mat image) {
	if (cv::imwrite(filename, image) == false) {
		std::cout << "imwrite error\n";
	};

	return 0;
}