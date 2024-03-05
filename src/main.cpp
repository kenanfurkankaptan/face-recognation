#include <fmt/core.h>
#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/core/utility.hpp>
#include <unordered_set>
#include <vector>

#include "face_detection/face_detection.h"
#include "face_recognation/face_recognation.h"
#include "include/constants.h"
#include "schemes/face_recognation_model_info.h"

using json = nlohmann::json;

namespace fs = std::filesystem;

int main(int argc, char** argv) {
	/** 0 = train model
	 * 1 = save images
	 * 2 = update model
	 * 3 = predict */
	int app_mode = -1;

	std::string label_name = "default";
	if (argc >= 2) {
		for (int i = 0; i < argc; i++) {
			if (strcmp(argv[i], "-train") == 0) {
				app_mode = 0;
			} else if (strcmp(argv[i], "-camera") == 0) {
				app_mode = 1;
			} else if (strcmp(argv[i], "-update") == 0) {
				app_mode = 2;
			} else if (strcmp(argv[i], "-predict") == 0) {
				app_mode = 3;
			} else if (strcmp(argv[i], "--label") == 0) {
				++i;
				label_name = argv[i];
			}
		}
	}

	if (app_mode == 0)
		fs::remove(Constants::Path::face_recognation_info);

	schemes::json_models::face_recognize::model_info info;
	if (fs::exists(Constants::Path::face_recognation_info)) {
		std::cout << "updating model" << std::endl;
		auto json = json::parse(std::ifstream(Constants::Path::face_recognation_info));
		schemes::json_models::face_recognize::from_json(json, info);
	} else {
		std::cout << "creating model" << std::endl;
	}

	if (app_mode == 0 || app_mode == 2) {
		std::unordered_set<std::string> labels;
		try {
			for (const auto& entry : fs::directory_iterator(Constants::Path::data)) {
				if (fs::is_directory(entry.status())) {
					labels.insert(entry.path().filename());
					std::cout << entry.path().filename() << std::endl;
				}
			}
		} catch (const std::filesystem::filesystem_error& e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}

		for (const auto& entry : info.labels) {
			std::cout << entry.name << std::endl;
			std::cout << labels.erase(entry.name) << std::endl;
		}

		std::vector<cv::Mat> image_mat;
		std::vector<int> image_labels;

		auto face_detector = face_detection(Constants::Path::face_detection, from_file);
		auto face_recog = face_recognizer(fmt::format("{}/trainer.xml", Constants::Path::face_recognation));

		int i = info.labels.back().id;
		for (const std::string& value : labels) {
			try {
				for (const auto& entry : fs::directory_iterator(fmt::format("{}/{}", Constants::Path::data, value))) {
					if (fs::is_regular_file(entry.status())) {
						cv::Mat img = face_detector.get_image(entry.path().generic_string());
						auto face_pos = face_detector.get_faces(img);
						auto faces = face_detector.crop_resize_faces(img, face_pos);
						image_mat.insert(
							image_mat.end(),
							std::make_move_iterator(faces.begin()),
							std::make_move_iterator(faces.end()));
					}
				}

				std::cout << "AaA: " << i << std::endl;

				info.labels.push_back({i, value});

				std::vector<int> temp(image_mat.size() - image_labels.size(), i++);
				image_labels.insert(image_labels.end(), temp.begin(), temp.end());

			} catch (const std::filesystem::filesystem_error& e) {
				std::cerr << "Error: " << e.what() << std::endl;
			}
		}

		try {
			std::ofstream outputFile(Constants::Path::face_recognation_info, std::ios_base::out);
			if (outputFile.is_open()) {
				json json_data;
				schemes::json_models::face_recognize::to_json(json_data, info);
				outputFile << std::setw(4) << json_data << std::endl;
				outputFile.close();
				std::cout << "JSON data saved to '" << Constants::Path::face_recognation_info << "'." << std::endl;
			} else {
				std::cerr << "Error opening file '" << Constants::Path::face_recognation_info << "' for writing." << std::endl;
			}
		} catch (const std::exception& e) {
			std::cerr << "Error saving JSON data: " << e.what() << std::endl;
		}

		face_recog.update(image_mat, image_labels);
		face_recog.save_model(fmt::format("{}/trainer.xml", Constants::Path::face_recognation));
	} else if (app_mode == 1) {
		std::cout << "take photo\n";
		std::cout << "Initiating face detector\n";

		try {
			if (fs::create_directory(fmt::format("{}/{}", Constants::Path::data, label_name)))
				std::cout << "Created a directory\n";
			else
				std::cerr << "Failed to create a directory\n";
		} catch (const std::exception& e) {
			std::cerr << e.what() << '\n';
		}

		auto face_detector = face_detection(Constants::Path::face_detection, from_camera);

		std::cout << "Capturing Camera\n";

		int counter = 0;
		while (true) {
			cv::Mat img = face_detector.get_image();
			auto face_pos = face_detector.get_faces(img);
			auto faces = face_detector.crop_resize_faces(img, face_pos);

			if (!faces.empty()) {
				counter++;
				if (counter % 25 == 0) {
					face_detector.save_image(fmt::format("{}/{}/{}.png", Constants::Path::data, label_name, counter), img);
				}
			}

			if (counter == 500) break;

			imshow("Image", img);
			cv::waitKey(1);
		}

	} else if (app_mode == 3) {
		std::cout << "predict\n";

		std::cout << "Loading Model\n";

		auto face_recog = face_recognizer();
		face_recog.load_model(fmt::format("{}/{}", Constants::Path::face_recognation, info.model));

		std::cout << "Initiating face detector\n";

		auto face_detector = face_detection(Constants::Path::face_detection, from_camera);

		std::cout << "Capturing Camera\n";

		while (true) {
			std::vector<face_predict_model> data;
			cv::Mat img = face_detector.get_image();
			auto face_pos = face_detector.get_faces(img);

			for (auto p : face_pos) {
				auto face = face_detector.crop_resize_faces(img, p);
				auto a = face_recog.predict(face);
				std::cout << "label: " << a.label << "\tconfidence: " << a.confidence << std::endl;

				std::string name = "error";

				for (const auto label : info.labels) {
					if (label.id == a.label) {
						name = label.name;
						break;
					}
				}

				face_detector.label_faces(p, name, img);
			}

			imshow("Image", img);
			cv::waitKey(1);
		}
	}

	return 0;
}
