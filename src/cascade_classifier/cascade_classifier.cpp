#include "cascade_classifier.h"

#include <filesystem>

namespace fs = std::filesystem;

cascade_classifier::cascade_classifier(std::string model_path) {
	std::cout << "Initiating CascadeClassifier Face Detector\n";

	if (model_path == "" || model_path.empty()) {
		std::cout << "face_detection::load_model: path is empty" << std::endl;
	}

	if (!fs::exists(model_path)) {
		std::cout << "face_detection::load_model: path is not exist -- " << model_path << std::endl;
	}
	if (!detector.load(model_path)) {
		std::cerr << "Error: Could not load Haar Cascade Classifier from " << model_path << std::endl;
	}
};

std::vector<cv::Rect> cascade_classifier::get_faces(cv::Mat img) {
	if (detector.empty()) {
		std::cerr << "cascade_classifier::load_model: cascade model does not exist\n";
		return {};
	}

	/** preprocess */
	cv::Mat img_grey;
	cvtColor(img, img_grey, cv::COLOR_BGRA2GRAY);

	std::vector<cv::Rect> faces;
	detector.detectMultiScale(img_grey, faces, 1.1, 10);

	return faces;
}
