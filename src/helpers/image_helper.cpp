#include "image_helper.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace image_helper {

	std::vector<cv::Mat> crop_resize_faces(cv::Mat image, std::vector<cv::Rect> pos) {
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

	cv::Mat crop_resize_faces(cv::Mat image, cv::Rect pos) {
		cv::Mat img_grey, temp;
		cvtColor(image, img_grey, cv::COLOR_BGRA2GRAY);
		cv::resize(img_grey(pos), temp, cv::Size(200, 200));

		return temp;
	}

	cv::Mat label_faces(cv::Rect pos, std::string name, cv::Mat image) {
		rectangle(image, pos.tl(), pos.br(), cv::Scalar(255, 0, 255), 3);
		putText(image, name, pos.tl(), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 69, 255), 3);

		return image;
	}

	int save_image(std::string filename, cv::Mat image) {
		if (cv::imwrite(filename, image) == false) {
			std::cout << "imwrite error\n";
		};

		return 0;
	}
}  // namespace image_helper
