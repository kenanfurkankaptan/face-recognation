#ifndef IMAGE_PROVIDER_H_
#define IMAGE_PROVIDER_H_

// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

enum source_type {
	from_file,
	from_camera
};

class image_provider {
   public:
	image_provider(source_type type, int api_preference = 0);
	cv::Mat get_image(std::string path = "");

   private:
	cv::VideoCapture cap;
	source_type type;
};

#endif /* IMAGE_PROVIDER_H_ */
