#include "yunet.h"

#include <filesystem>

namespace fs = std::filesystem;

yunet::yunet(std::string model_path) {
	std::cout << "Initiating YuNet Face Detector\n";

	if (model_path == "" || model_path.empty()) {
		std::cout << "face_detection::load_model: path is empty" << std::endl;
	}

	if (!fs::exists(model_path)) {
		std::cout << "face_detection::load_model: path is not exist -- " << model_path << std::endl;
	}

	detector = cv::FaceDetectorYN::create(model_path, "", cv::Size(1280, 720));

	if (detector.empty()) {
		std::cerr << "cascade_classifier::load_model: cascade model does not exist\n";
	}
}

std::vector<cv::Rect> yunet::get_faces(cv::Mat img) {
	if (detector.empty()) {
		std::cerr << "cascade_classifier::load_model: cascade model does not exist\n";
		return {};
	}

	cv::Mat data;
	std::vector<cv::Rect> faces;
	detector->detect(img, data);

	/** TODO: handle
	 *  terminate called after throwing an instance of 'cv::Exception'
	 *  what():  OpenCV(4.5.4) ./modules/core/src/matrix.cpp:810: error: (-215:Assertion failed)
	 *  0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows in function 'Mat'
	 */

	for (int i = 0; i < data.rows; i++) {
		// Print results
		// std::cout << "Face " << i
		// 		  << ", top-left coordinates: (" << data.at<float>(i, 0) << ", " << data.at<float>(i, 1) << "), "
		// 		  << "box width: " << data.at<float>(i, 2) << ", box height: " << data.at<float>(i, 3) << ", "
		// 		  << "score: " << cv::format("%.2f", data.at<float>(i, 14))
		// 		  << std::endl;

		// // Draw bounding box
		// cv::rectangle(img, cv::Rect2i(int(data.at<float>(i, 0)), int(data.at<float>(i, 1)), int(data.at<float>(i, 2)), int(data.at<float>(i, 3))), cv::Scalar(0, 255, 0), 2);

		faces.push_back(cv::Rect2i(int(data.at<float>(i, 0)), int(data.at<float>(i, 1)), int(data.at<float>(i, 2)), int(data.at<float>(i, 3))));

		// 	// Draw landmarks
		// 	cv::circle(img, cv::Point2i(int(data.at<float>(i, 4)), int(data.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), 2);
		// 	cv::circle(img, cv::Point2i(int(data.at<float>(i, 6)), int(data.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), 2);
		// 	cv::circle(img, cv::Point2i(int(data.at<float>(i, 8)), int(data.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), 2);
		// 	cv::circle(img, cv::Point2i(int(data.at<float>(i, 10)), int(data.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), 2);
		// 	cv::circle(img, cv::Point2i(int(data.at<float>(i, 12)), int(data.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), 2);
	}

	return faces;
}
