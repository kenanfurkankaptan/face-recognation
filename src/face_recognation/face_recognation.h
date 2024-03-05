#ifndef FACE_RECOGNIZER_H_
#define FACE_RECOGNIZER_H_

#include <iostream>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

#include "../schemes/face_predict_model.h"

class face_recognizer {
   public:
	face_recognizer(std::string path = "");
	int train(std::vector<cv::Mat> images, int label);
	int train(std::vector<cv::Mat> images, std::vector<int> label);

	int update(std::vector<cv::Mat> images, int label);
	int update(std::vector<cv::Mat> images, std::vector<int> label);

	face_predict_model predict(cv::Mat faceROI);
	int save_model(std::string path);
	int load_model(std::string path);

   private:
	cv::Ptr<cv::face::FaceRecognizer> recognizer;
};

#endif /* FACE_RECOGNIZER_H_ */
