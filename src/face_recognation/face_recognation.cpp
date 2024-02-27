#include "face_recognation.h"

face_recognizer::face_recognizer() {
	recognizer = cv::face::LBPHFaceRecognizer::create();

	if (recognizer.empty()) {
		std::cerr << "face_recognizer::face_recognizer: LBPHFaceRecognizer empty\n";
	}
}

int face_recognizer::train(std::vector<cv::Mat> images, int label) {
	std::vector<int> label_vector(images.size(), label);

	if (images.empty() || label_vector.empty() || images.size() != label_vector.size()) {
		std::cerr << "Error: Empty or mismatched dataset." << std::endl;
		return -1;
	}

	try {
		recognizer->train(images, label_vector);
	} catch (const cv::Exception& e) {
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
	}

	return 0;
}

face_predict_model face_recognizer::predict(cv::Mat faceROI) {
	int label;
	double confidence;
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
