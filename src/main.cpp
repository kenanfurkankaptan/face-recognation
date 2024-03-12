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

#include "LBPH_face_recognizer/LBPH_face_recognizer.h"
#include "cascade_classifier/cascade_classifier.h"
#include "helpers/image_helper.h"
#include "helpers/image_provider.h"
#include "include/constants.h"
#include "schemes/face_recognation_model_info.h"
#include "yunet/yunet.h"

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

	IFaceDetection* face_detector = nullptr;

	face_detector = new yunet("models/face detection/face_detection_yunet_2022mar.onnx");
	// face_detector = new cascade_classifier(Constants::Path::face_detection);

	// auto image = image_provider(from_file);

	// double brightness = 0;
	// double gamma = 0;
	// double saturation = 0;

	// for (const auto& entry : fs::directory_iterator(fmt::format("{}/foreign", Constants::Path::data))) {
	// 	if (fs::is_regular_file(entry.status())) {
	// 		cv::Mat img = image.get_image(entry.path().generic_string());

	// 		double t_brightness = 0;
	// 		double t_gamma = 0;
	// 		double t_saturation = 0;

	// 		auto face_pos = face_detector->get_faces(img);
	// 		auto faces = image_helper::crop_resize_faces(img, face_pos);

	// 		for (auto i : faces) {
	// 			image_helper::calculateBrightnessGammaSaturation(i, t_brightness, t_gamma, t_saturation);

	// 			brightness += t_brightness;
	// 			gamma += t_gamma;
	// 			saturation += t_saturation;
	// 		}
	// 	}
	// }

	// brightness = brightness / 450;
	// gamma = gamma / 450;
	// saturation = saturation / 450;

	// std::cout << fmt::format("brightness: {} -- gamma: {} -- saturation: {}", brightness, gamma, saturation) << std::endl;

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
		auto image = image_provider(from_file);

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

		auto face_recog = LBPH_face_recognizer();
		if (!info.model.empty()) face_recog.load_model(fmt::format("{}/trainer.xml", Constants::Path::face_recognation));

		/** vector.back gives segmentation fault if vector is empty */
		int i = info.labels.size() == 0 ? 0 : info.labels.back().id + 1;

		for (const std::string& value : labels) {
			try {
				for (const auto& entry : fs::directory_iterator(fmt::format("{}/{}", Constants::Path::data, value))) {
					if (fs::is_regular_file(entry.status())) {
						cv::Mat img = image.get_image(entry.path().generic_string());
						auto face_pos = face_detector->get_faces(img);

						for (auto face_rect : face_pos) {
							auto face = image_helper::crop_resize_faces(img, face_rect);

							if (face.empty()) continue;

							auto current_bgs = image_helper::calculate_bgs(face);
							auto adjusted_face = image_helper::preprocess_image(face, current_bgs, true);
							image_mat.push_back(adjusted_face);
						}
					}
				}

				info.labels.push_back({i, value});

				std::vector<int> temp(image_mat.size() - image_labels.size(), i++);
				image_labels.insert(image_labels.end(), temp.begin(), temp.end());

			} catch (const std::filesystem::filesystem_error& e) {
				std::cerr << "Error: " << e.what() << std::endl;
			}
		}

		info.model = "trainer.xml";

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

		face_recog.save_model(fmt::format("{}/{}", Constants::Path::face_recognation, info.model));

	} else if (app_mode == 1) {
		auto image = image_provider(from_camera);

		try {
			if (fs::create_directory(fmt::format("{}/{}", Constants::Path::data, label_name)))
				std::cout << "Created a directory\n";
			else
				std::cerr << "Failed to create a directory\n";
		} catch (const std::exception& e) {
			std::cerr << e.what() << '\n';
		}

		std::cout << "Capturing Camera\n";
		int counter = 0;
		while (true) {
			cv::Mat img = image.get_image();
			auto face_pos = face_detector->get_faces(img);
			auto faces = image_helper::crop_resize_faces(img, face_pos);

			if (!faces.empty()) {
				counter++;
				if (counter % 10 == 0) {
					image_helper::save_image(fmt::format("{}/{}/{}.png", Constants::Path::data, label_name, counter), img);
				}
			}

			if (counter == 200) break;

			imshow("Image", img);
			cv::waitKey(1);
		}

	} else if (app_mode == 3) {
		auto image = image_provider(from_camera);
		auto face_recog = LBPH_face_recognizer(fmt::format("{}/{}", Constants::Path::face_recognation, info.model));

		std::cout << "Capturing Camera\n";
		while (true) {
			std::vector<face_predict_model> data;
			cv::Mat adjusted_face;
			cv::Mat adjusted_image;

			cv::Mat img = image.get_image();
			auto face_pos = face_detector->get_faces(img);
			for (auto p : face_pos) {
				auto face = image_helper::crop_resize_faces(img, p);

				if (face.empty()) continue;

				auto current_bgs = image_helper::calculate_bgs(face);
				adjusted_face = image_helper::preprocess_image(face, current_bgs, true);
				adjusted_image = image_helper::preprocess_image(img, current_bgs);

				auto prediction_result = face_recog.predict(adjusted_face);
				std::cout << "label: " << prediction_result.label << "\tconfidence: " << prediction_result.confidence << std::endl;

				std::string name = "error";

				for (const auto label : info.labels) {
					if (label.id == prediction_result.label) {
						name = label.name;
						break;
					}
				}

				image_helper::label_faces(p, name, adjusted_image);
			}

			imshow("Image", adjusted_image.empty() ? img : adjusted_image);
			cv::waitKey(1);
		}
	}

	return 0;
}
