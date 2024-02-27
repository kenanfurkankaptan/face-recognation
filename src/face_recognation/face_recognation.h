
#include <iostream>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

#include "../schemes/face_predict_model.h"

class face_recognizer {
   public:
	face_recognizer();
	int train(std::vector<cv::Mat> images, int label);
	face_predict_model predict(cv::Mat faceROI);
	int save_model(std::string path);
	int load_model(std::string path);

   private:
	cv::Ptr<cv::face::FaceRecognizer> recognizer;
};