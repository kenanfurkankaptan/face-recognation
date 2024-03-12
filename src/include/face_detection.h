#ifndef FACE_DETECTION_H_
#define FACE_DETECTION_H_

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

class IFaceDetection {
   public:
	IFaceDetection(){};
	IFaceDetection(std::string model_path){};

	virtual std::vector<cv::RotatedRect> get_faces(cv::Mat image) = 0;
};

#endif /* FACE_DETECTION_H_ */
