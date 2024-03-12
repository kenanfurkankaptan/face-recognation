#ifndef YUNET_H_
#define YUNET_H_

#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "../include/face_detection.h"

/** https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html */

class yunet : public IFaceDetection {
   public:
	yunet() = delete;
	yunet(std::string model_path);

	std::vector<cv::RotatedRect> get_faces(cv::Mat image);

   private:
	cv::Ptr<cv::FaceDetectorYN> detector;

	void set_size(cv::Size img_size);
	cv::Size current_size = cv::Size(1280, 720);
};

#endif /* YUNET_H_ */
