#include "yunet.h"

#include <fmt/core.h>

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

void yunet::set_size(cv::Size img_size) {
	if (this->current_size != img_size) {
		detector->setInputSize(img_size);
		current_size = img_size;
	}
}

std::vector<cv::RotatedRect> yunet::get_faces(cv::Mat img) {
	if (detector.empty()) {
		std::cerr << "cascade_classifier::load_model: cascade model does not exist\n";
		return {};
	}

	set_size(img.size());

	cv::Mat data;
	std::vector<cv::RotatedRect> faces;
	detector->detect(img, data);
	for (int i = 0; i < data.rows; i++) {
		// Print results
		// std::cout << "Face " << i
		// 		  << ", top-left coordinates: (" << data.at<float>(i, 0) << ", " << data.at<float>(i, 1) << "), "
		// 		  << "box width: " << data.at<float>(i, 2) << ", box height: " << data.at<float>(i, 3) << ", "
		// 		  << "score: " << cv::format("%.2f", data.at<float>(i, 14))
		// 		  << std::endl;

		// // Draw bounding box
		// cv::rectangle(img, cv::Rect2i(int(data.at<float>(i, 0)), int(data.at<float>(i, 1)), int(data.at<float>(i, 2)), int(data.at<float>(i, 3))), cv::Scalar(0, 255, 0), 2);

		// // Draw landmarks
		// cv::circle(img, cv::Point2i(int(data.at<float>(i, 4)), int(data.at<float>(i, 5))), 2, cv::Scalar(255, 0, 0), 2);
		// cv::circle(img, cv::Point2i(int(data.at<float>(i, 6)), int(data.at<float>(i, 7))), 2, cv::Scalar(0, 0, 255), 2);
		// cv::circle(img, cv::Point2i(int(data.at<float>(i, 8)), int(data.at<float>(i, 9))), 2, cv::Scalar(0, 255, 0), 2);
		// cv::circle(img, cv::Point2i(int(data.at<float>(i, 10)), int(data.at<float>(i, 11))), 2, cv::Scalar(255, 0, 255), 2);
		// cv::circle(img, cv::Point2i(int(data.at<float>(i, 12)), int(data.at<float>(i, 13))), 2, cv::Scalar(0, 255, 255), 2);

		// double angle1 = atan2(data.at<float>(i, 5) - data.at<float>(i, 7), data.at<float>(i, 4) - data.at<float>(i, 6));
		// double angle2 = atan2(data.at<float>(i, 11) - data.at<float>(i, 13), data.at<float>(i, 10) - data.at<float>(i, 12));

		double angle1 = atan2(data.at<float>(i, 7) - data.at<float>(i, 5), data.at<float>(i, 6) - data.at<float>(i, 4));
		double angle2 = atan2(data.at<float>(i, 13) - data.at<float>(i, 11), data.at<float>(i, 12) - data.at<float>(i, 10));

		angle1 = angle1 * 180.0 / CV_PI;
		angle2 = angle2 * 180.0 / CV_PI;

		// std::cout << fmt::format("angle1: {} -- angle2: {}\n", angle1, angle2);

		auto original_rect = cv::Rect2i(int(data.at<float>(i, 0)), int(data.at<float>(i, 1)), int(data.at<float>(i, 2)), int(data.at<float>(i, 3)));
		cv::RotatedRect rotated_rect = cv::RotatedRect(
			cv::Point2f(original_rect.x + original_rect.width / 2.0, original_rect.y + original_rect.height / 2.0),
			cv::Size2f(original_rect.width, original_rect.height),
			(angle1 + angle2) / 2  // Rotation angle: 30 degrees
		);

		faces.push_back(rotated_rect);
	}

	// imshow("Image", img);
	// cv::waitKey(1);

	return faces;
}
