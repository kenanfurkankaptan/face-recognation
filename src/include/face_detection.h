#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class face_detection {
   public:
	face_detection() = default;
	face_detection(std::string path){};
	virtual int load_model(std::string path) = 0;
	virtual int select_source(std::string path) = 0;
	virtual std::vector<cv::Mat> get_faces(std::string path) = 0;
};