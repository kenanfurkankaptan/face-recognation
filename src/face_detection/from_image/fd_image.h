#ifndef FD_IMAGE_H_
#define FD_IMAGE_H_

#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

#include "../../include/face_detection.h"

class fd_image : public face_detection {
   public:
	fd_image(std::string path) {
		if (path == "" || path.empty()) {
			std::cout << "fd_image::load_model: path is empty" << std::endl;
		}

		if (!fs::exists(path)) {
			std::cout << "fd_image::load_model: path is not exist -- " << path << std::endl;
		}
		if (!face_cascade.load(path)) {
			std::cerr << "Error: Could not load Haar Cascade Classifier from " << path << std::endl;
		}
	};
	std::vector<cv::Mat> get_faces(std::string path);

   private:
	cv::CascadeClassifier face_cascade;
	cv::Mat img;
};

#endif /* FD_IMAGE_H_ */
