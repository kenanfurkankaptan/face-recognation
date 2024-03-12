#include "LBPH_face_recognizer.h"

#include <filesystem>

namespace fs = std::filesystem;

LBPH_face_recognizer::LBPH_face_recognizer(std::string path) {
	std::cout << "Creating LBPHFaceRecognizer model" << std::endl;
	recognizer = cv::face::LBPHFaceRecognizer::create(2);

	if (!path.empty() && fs::exists(path)) {
		std::cout << "Loading trained LBPHFaceRecognizer model" << std::endl;
		recognizer->read(path);
	}

	if (recognizer.empty()) {
		std::cerr << "face_recognizer::face_recognizer: LBPHFaceRecognizer empty\n";
	}
}

int LBPH_face_recognizer::train(std::vector<cv::Mat> images, int label) {
	std::vector<int> label_vector(images.size(), label);

	return this->train(images, label_vector);
}

int LBPH_face_recognizer::train(std::vector<cv::Mat> images, std::vector<int> label) {
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

int LBPH_face_recognizer::update(std::vector<cv::Mat> images, int label) {
	std::vector<int> label_vector(images.size(), label);

	return this->update(images, label_vector);
}

int LBPH_face_recognizer::update(std::vector<cv::Mat> images, std::vector<int> label) {
	std::cout << "Updating LBPHFaceRecognizer model" << std::endl;

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

face_predict_model LBPH_face_recognizer::predict(cv::Mat face_ROI) {
	int label = -1;
	double confidence = -1;
	recognizer->predict(face_ROI, label, confidence);

	return {label, confidence};
}

int LBPH_face_recognizer::save_model(std::string path) {
	std::cout << "Saving LBPHFaceRecognizer model" << std::endl;

	try {
		recognizer->save(path);
	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}

	return 0;
}

int LBPH_face_recognizer::load_model(std::string path) {
	std::cout << "Loading trained LBPHFaceRecognizer model" << std::endl;

	try {
		recognizer->read(path);
	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}

	return 0;
}
