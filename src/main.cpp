#include <string.h>

#include <filesystem>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <vector>

#include "face_detection/face_detection.h"
#include "face_recognation/face_recognation.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
	cv::setNumThreads(16);

	/** 0 = train model
	 * 1 =
	 * 2 = predict */
	int app_mode = 1;

	int label = -1;
	if (argc >= 2) {
		for (int i = 0; i < argc; i++) {
			if (strcmp(argv[i], "-train") == 0) {
				app_mode = 0;
			} else if (strcmp(argv[i], "-train2") == 0) {
				app_mode = 1;
			} else if (strcmp(argv[i], "-predict") == 0) {
				app_mode = 2;
			} else if (strcmp(argv[i], "--label") == 0) {
				++i;
				label = std::system(argv[i]);
			}
		}
	}

	if (app_mode == 0) {
		std::cout << "train\n";

		std::vector<cv::Mat> image_mat;
		auto face_detector = face_detection("resources/models/haarcascade_frontalface_default.xml", from_file);

		try {
			for (const auto& entry : fs::directory_iterator("./resources/dataset/random")) {
				if (fs::is_regular_file(entry.path())) {
					cv::Mat img = face_detector.get_image(entry.path().generic_string());
					auto face_pos = face_detector.get_faces(img);
					auto faces = face_detector.crop_faces(img, face_pos);
					image_mat.insert(
						image_mat.end(),
						std::make_move_iterator(faces.begin()),
						std::make_move_iterator(faces.end()));
				}
			}
		} catch (const std::filesystem::filesystem_error& e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}

		auto face_recog = face_recognizer();
		face_recog.train(image_mat, 0);
		face_recog.save_model("resources/models/trainer.xml");
	}

	else if (app_mode == 1) {
		std::cout << "train2\n";

		std::cout << "Loading Model\n";

		auto face_recog = face_recognizer("resources/models/trainer.xml");

		std::cout << "Initiating face detector\n";

		auto face_detector = face_detection("resources/models/haarcascade_frontalface_default.xml", from_camera);

		std::cout << "Capturing Camera\n";

		int counter = 0;
		while (true) {
			cv::Mat img = face_detector.get_image();
			auto face_pos = face_detector.get_faces(img);
			auto faces = face_detector.crop_faces(img, face_pos);

			if (!faces.empty()) counter++;

			face_recog.update(faces, 1);

			if (counter == 4) break;

			cv::waitKey(5);
		}

		face_recog.save_model("resources/models/trainer2.xml");
	} else {
		std::cout << "predict\n";

		std::cout << "Loading Model\n";

		auto face_recog = face_recognizer();
		face_recog.load_model("resources/models/trainer2.xml");

		std::cout << "Initiating face detector\n";

		auto face_detector = face_detection("resources/models/haarcascade_frontalface_default.xml", from_camera);

		std::cout << "Capturing Camera\n";

		while (true) {
			std::vector<face_predict_model> data;
			cv::Mat img = face_detector.get_image();
			auto face_pos = face_detector.get_faces(img);

			for (auto p : face_pos) {
				auto face = face_detector.crop_faces(img, p);
				auto a = face_recog.predict(face);
				std::cout << "label: " << a.label << "\tconfidence: " << a.confidence << std::endl;

				std::string name;
				switch (a.label) {
					case 0:
						name = "foreign";
						break;
					case 1:
						name = "kfk";

						break;
					case 2:
						name = "cut";

						break;
					default:
						name = "foreign";
				}
				face_detector.label_faces(p, name, img);
			}

			imshow("Image", img);
			cv::waitKey(1);
		}
	}

	return 0;
}
