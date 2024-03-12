#ifndef FACE_RECOGNATION_H_
#define FACE_RECOGNATION_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../schemes/face_predict_model.h"

class IFaceRecognation {
   public:
	IFaceRecognation(std::string path = ""){};
	virtual int train(std::vector<cv::Mat> images, int label) = 0;
	virtual int train(std::vector<cv::Mat> images, std::vector<int> label) = 0;

	virtual int update(std::vector<cv::Mat> images, int label) = 0;
	virtual int update(std::vector<cv::Mat> images, std::vector<int> label) = 0;

	virtual face_predict_model predict(cv::Mat face_ROI) = 0;
	virtual int save_model(std::string path) = 0;
	virtual int load_model(std::string path) = 0;
};

#endif /* FACE_RECOGNATION_H_ */
