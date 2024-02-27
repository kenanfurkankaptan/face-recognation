#ifndef INCLUDE_FACE_DETECTION_H_
#define INCLUDE_FACE_DETECTION_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class face_detection {
   public:
	face_detection() = default;
	face_detection(std::string path){};
	virtual std::vector<cv::Mat> get_faces(std::string path) = 0;
};

#endif /* INCLUDE_FACE_DETECTION_H_ */
