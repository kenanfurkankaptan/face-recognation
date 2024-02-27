#include <string.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "face_recognation/face_recognation.h"
#include "fd_video/fd_video.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
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
		auto face_detector = fd_video("resources/models/haarcascade_frontalface_default.xml");

		try {
			for (const auto& entry : fs::directory_iterator("./resources/dataset/random")) {
				if (fs::is_regular_file(entry.path())) {
					auto image = face_detector.get_faces(entry.path().generic_string());
					image_mat.insert(
						image_mat.end(),
						std::make_move_iterator(image.begin()),
						std::make_move_iterator(image.end()));
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

		auto face_recog = face_recognizer();
		face_recog.load_model("resources/models/trainer.xml");

		auto face_detector = fd_video("resources/models/haarcascade_frontalface_default.xml");
		face_detector.select_source("");

		std::cout << "Capturing Camera\n";

		int counter = 0;
		while (true) {
			auto images = face_detector.get_faces("");

			if (!images.empty()) counter++;

			face_recog.train(images, 1);

			if (counter == 40) break;

			cv::waitKey(1);
		}

	}

	else {
		std::cout << "predict\n";

		auto face_recog = face_recognizer();
		face_recog.load_model("resources/models/trainer.xml");

		auto face_detector = fd_video("resources/models/haarcascade_frontalface_default.xml");
		face_detector.select_source("");

		std::vector<face_predict_model> data;
		auto images = face_detector.get_faces("");
		for (auto i : images) {
			data.push_back(face_recog.predict(i));
		}
	}

	return 0;
}
