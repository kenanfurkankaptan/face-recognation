#ifndef CASCADE_CLASSIFIER_H_
#define CASCADE_CLASSIFIER_H_

#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include "../include/face_detection.h"

class cascade_classifier : public IFaceDetection {
   public:
	cascade_classifier() = delete;
	cascade_classifier(std::string model_path);

	std::vector<cv::Rect> get_faces(cv::Mat image);

   private:
	cv::CascadeClassifier detector;
};

#endif /* CASCADE_CLASSIFIER_H_ */
