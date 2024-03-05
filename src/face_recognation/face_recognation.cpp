#include "face_recognation.h"

#include <filesystem>

namespace fs = std::filesystem;

face_recognizer::face_recognizer(std::string path) {
	std::cout << "Creating model" << std::endl;
	recognizer = cv::face::LBPHFaceRecognizer::create();

	if (!path.empty() && fs::exists(path)) {
		std::cout << "Loading trained model" << std::endl;
		recognizer->read(path);
	}

	if (recognizer.empty()) {
		std::cerr << "face_recognizer::face_recognizer: LBPHFaceRecognizer empty\n";
	}
}

int face_recognizer::train(std::vector<cv::Mat> images, int label) {
	std::vector<int> label_vector(images.size(), label);

	return this->train(images, label_vector);
}

int face_recognizer::train(std::vector<cv::Mat> images, std::vector<int> label) {
	if (images.empty() || label.empty() || images.size() != label.size()) {
		std::cerr << "Error: size: " << images.size() << std::endl;

		std::cerr << "Error: Empty or mismatched dataset." << std::endl;
		return -1;
	}

	try {
		recognizer->train(images, label);
	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}

	return 0;
}

int face_recognizer::update(std::vector<cv::Mat> images, int label) {
	std::vector<int> label_vector(images.size(), label);

	return this->update(images, label_vector);
}

int face_recognizer::update(std::vector<cv::Mat> images, std::vector<int> label) {
	if (images.empty() || label.empty() || images.size() != label.size()) {
		std::cerr << "Error: size: " << images.size() << std::endl;

		std::cerr << "Error: Empty or mismatched dataset." << std::endl;
		return -1;
	}

	try {
		recognizer->update(images, label);
	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}

	return 0;
}

face_predict_model face_recognizer::predict(cv::Mat faceROI) {
	int label = -1;
	double confidence = -1;
	recognizer->predict(faceROI, label, confidence);

	return {label, confidence};
}

int face_recognizer::save_model(std::string path) {
	recognizer->save(path);
	return 0;
}

int face_recognizer::load_model(std::string path) {
	recognizer->read(path);
	return 0;
}
