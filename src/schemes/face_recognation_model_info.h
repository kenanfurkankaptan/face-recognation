//  To parse this JSON data, first install
//
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Detection data = nlohmann::json::parse(jsonString);

#pragma once

#include <nlohmann/json.hpp>

namespace schemes {
	namespace json_models {
		namespace face_recognize {
			using nlohmann::json;

#ifndef NLOHMANN_UNTYPED_face_recognize_info_HELPER
#define NLOHMANN_UNTYPED_face_recognize_info_HELPER
			inline json get_untyped(const json& j, const char* property) {
				if (j.find(property) != j.end()) {
					return j.at(property).get<json>();
				}
				return json();
			}

			inline json get_untyped(const json& j, std::string property) {
				return get_untyped(j, property.data());
			}
#endif

			struct label {
				int32_t id;
				std::string name;
			};

			struct model_info {
				std::string model = "";
				std::vector<label> labels;
			};
		}  // namespace face_recognize

		namespace face_recognize {
			void from_json(const json& j, label& x);
			void to_json(json& j, const label& x);

			void from_json(const json& j, model_info& x);
			void to_json(json& j, const model_info& x);

			inline void from_json(const json& j, label& x) {
				x.id = j.at("id").get<int64_t>();
				x.name = j.at("name").get<std::string>();
			}

			inline void to_json(json& j, const label& x) {
				j = json::object();
				j["id"] = x.id;
				j["name"] = x.name;
			}

			inline void from_json(const json& j, model_info& x) {
				x.model = j.at("model").get<std::string>();
				x.labels = j.at("labels").get<std::vector<label>>();
			}

			inline void to_json(json& j, const model_info& x) {
				j = json::object();
				j["model"] = x.model;
				j["labels"] = x.labels;
			}
		}  // namespace face_recognize
	}	   // namespace json_models
}  // namespace schemes
