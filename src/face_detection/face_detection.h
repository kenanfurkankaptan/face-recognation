#ifndef FACE_DETECTION_H_
#define FACE_DETECTION_H_

#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

enum source_type {
	from_file,
	from_camera
};

class face_detection {
   public:
	face_detection(std::string model_path, source_type type);

	cv::Mat get_image(std::string path = "");
	std::vector<cv::Rect> get_faces(cv::Mat image);
	std::vector<cv::Mat> crop_resize_faces(cv::Mat image, std::vector<cv::Rect>);
	cv::Mat crop_resize_faces(cv::Mat image, cv::Rect pos);

	int save_image(std::string filename, cv::Mat image);
	cv::Mat label_faces(cv::Rect pos, std::string name, cv::Mat image);

   private:
	cv::CascadeClassifier face_cascade;
	cv::VideoCapture cap;
	source_type type;
};

#endif /* FACE_DETECTION_H_ */
