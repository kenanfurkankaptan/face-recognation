#include "image_helper.h"

#include <fmt/core.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace image_helper {

	std::vector<cv::Mat> crop_resize_faces(cv::Mat image, std::vector<cv::RotatedRect> pos) {
		std::vector<cv::Mat> faces;
		cv::Mat img_grey;
		cvtColor(image, img_grey, cv::COLOR_BGRA2GRAY);

		for (auto roi : pos) {
			cv::Mat temp;
			auto bounding_rect = roi.boundingRect();

			if (bounding_rect.x >= 0 && bounding_rect.y >= 0 && bounding_rect.width >= 0 && bounding_rect.height >= 0 && bounding_rect.x + bounding_rect.width <= image.cols && bounding_rect.y + bounding_rect.height <= image.rows) {
				auto cropped = crop_resize_faces(image, roi);
				if (!cropped.empty())
					faces.push_back(cropped);
			} else {
				std::cout << "warning: Assertion failed: 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows\n";
			}
		}

		return faces;
	}

	cv::Mat crop_resize_faces(cv::Mat image, cv::RotatedRect pos) {
		cv::Mat img_rotated, temp;
		auto bounding_rect = pos.boundingRect();

		if (bounding_rect.x >= 0 && bounding_rect.y >= 0 && bounding_rect.width >= 0 && bounding_rect.height >= 0 && bounding_rect.x + bounding_rect.width <= image.cols && bounding_rect.y + bounding_rect.height <= image.rows) {
		} else {
			std::cout << "warning: Assertion failed: 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows\n";
			/** return empty image */
			return temp;
		}

		cv::Mat rotation_matrix = cv::getRotationMatrix2D(pos.center, pos.angle, 1.0);
		cv::warpAffine(image, img_rotated, rotation_matrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

		// Clip the rotated image with the bounding box
		temp = img_rotated(pos.boundingRect());
		cv::resize(temp, temp, cv::Size(200, 200));

		return temp;
	}

	cv::Mat label_faces(cv::RotatedRect pos, std::string name, cv::Mat image) {
		auto bounding_rect = pos.boundingRect();

		rectangle(image, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(255, 0, 255), 3);
		putText(image, name, bounding_rect.tl(), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 69, 255), 3);

		return image;
	}

	int save_image(std::string filename, cv::Mat image) {
		if (cv::imwrite(filename, image) == false) {
			std::cout << "imwrite error\n";
		};

		return 0;
	}

	bgs calculate_bgs(cv::Mat& image) {
		cv::Mat hsv_image;
		cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

		cv::Mat hist;
		int histSize[] = {8, 8, 8};
		float range[] = {0, 256};
		const float* histRange[] = {range, range, range};
		int channels[3] = {0, 1, 2};
		cv::calcHist(&hsv_image, 1, channels, cv::Mat(), hist, 3, histSize, histRange, true, false);

		cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		float brightness = 0;
		float gamma = 0;
		float saturation = 0;

		for (int i = 0; i < histSize[0]; ++i) {
			for (int j = 0; j < histSize[1]; ++j) {
				for (int k = 0; k < histSize[2]; ++k) {
					brightness += hist.at<float>(i, j, k) * (i + j + k);
					gamma += hist.at<float>(i, j, k);
					saturation += hist.at<float>(i, j, k) * cv::log(hist.at<float>(i, j, k) + 1e-10);
				}
			}
		}
		// std::cout << fmt::format("brightness: {} -- gamma: {} -- saturation: {}", brightness, gamma, saturation) << std::endl;

		return {brightness, gamma, saturation};
	}

	cv::Mat preprocess_image(cv::Mat& image, bgs values, bool gray_image) {
		cv::Mat hsv_image, adjusted_image;
		cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

		float target_brightness = 45.0;
		float target_gamma = 5.0;
		float target_saturation = -6;

		double brightnessRatio = target_brightness / values.brightness;
		double gammaRatio = target_gamma / values.gamma;
		double saturationRatio = target_saturation / values.saturation;

		hsv_image.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) -> void {
			pixel[2] = cv::saturate_cast<uchar>(pixel[2] * brightnessRatio);
			pixel[2] = cv::saturate_cast<uchar>(std::pow((pixel[2] / 255.0), gammaRatio) * 255.0);
			pixel[1] = cv::saturate_cast<uchar>(pixel[1] * saturationRatio);
		});

		cvtColor(hsv_image, adjusted_image, cv::COLOR_HSV2BGR);

		if (gray_image) cvtColor(adjusted_image, adjusted_image, cv::COLOR_BGRA2GRAY);

		return adjusted_image;
	}

}  // namespace image_helper
