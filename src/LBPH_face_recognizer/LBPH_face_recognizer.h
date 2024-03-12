#ifndef LBPH_FACE_RECOGNIZER_H_
#define LBPH_FACE_RECOGNIZER_H_

#include <iostream>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

#include "../include/face_recognation.h"
#include "../schemes/face_predict_model.h"

class LBPH_face_recognizer : IFaceRecognation {
   public:
	LBPH_face_recognizer(std::string path = "");
	int train(std::vector<cv::Mat> images, int label);
	int train(std::vector<cv::Mat> images, std::vector<int> label);

	int update(std::vector<cv::Mat> images, int label);
	int update(std::vector<cv::Mat> images, std::vector<int> label);

	face_predict_model predict(cv::Mat face_ROI);
	int save_model(std::string path);
	int load_model(std::string path);

   private:
	cv::Ptr<cv::face::FaceRecognizer> recognizer;
};

#endif /* LBPH_FACE_RECOGNIZER_H_ */
