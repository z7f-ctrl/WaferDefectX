#include "defect_localization.hpp"
#include "feature_extraction.hpp"
#include "onnx_classifier.hpp"
#include "preprocess.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

static void log_info(const std::string &msg) {
  std::cout << "[INFO] " << msg << std::endl;
}

static void log_error(const std::string &msg) {
  std::cerr << "[ERROR] " << msg << std::endl;
}

static std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':  out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n";  break;
    default:   out += c;
    }
  }
  return out;
}

static std::string features_to_json(const DefectFeatures &f) {
  std::ostringstream ss;
  ss << "[";
  for (int i = 0; i < FEATURE_DIM; ++i) {
    if (i > 0) ss << ", ";
    ss << f.values[i];
  }
  ss << "]";
  return ss.str();
}

static void validate_input(const cv::Mat &image, const std::string &path) {
  if (image.empty()) {
    throw std::runtime_error("Could not open or find the image at " + path);
  }
  if (image.rows < 2 || image.cols < 2) {
    throw std::invalid_argument("Image too small (" + std::to_string(image.cols) +
                                "x" + std::to_string(image.rows) +
                                "). Minimum is 2x2.");
  }
  if (image.channels() != 1 && image.channels() != 3) {
    throw std::invalid_argument("Expected 1 or 3 channels, got " +
                                std::to_string(image.channels()));
  }
}

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      log_error("Usage: ./WaferDefectX_Run [--json] [--model model.onnx] <image_path>");
      return -1;
    }

    bool json_output = false;
    std::string image_path;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--json") {
        json_output = true;
      } else if (std::string(argv[i]) == "--model") {
        if (i + 1 < argc) model_path = argv[++i];
      } else {
        image_path = argv[i];
      }
    }

    if (image_path.empty()) {
      log_error("Usage: ./WaferDefectX_Run [--json] <image_path>");
      return -1;
    }

    log_info("Starting WaferDefectX C++ Pipeline");
    log_info("Loading image: " + image_path);

    cv::Mat image = cv::imread(image_path);
    validate_input(image, image_path);

    log_info("Image loaded. Resolution: " + std::to_string(image.cols) +
             "x" + std::to_string(image.rows) +
             " Channels: " + std::to_string(image.channels()));

    Preprocessor preprocessor;
    DefectLocalizer localizer;
    FeatureExtractor extractor;

    std::unique_ptr<OnnxClassifier> classifier;
    if (!model_path.empty()) {
      log_info("Loading ONNX model: " + model_path);
      classifier = std::make_unique<OnnxClassifier>(model_path);
      log_info("Model loaded successfully");
    }

    auto start = std::chrono::high_resolution_clock::now();

    log_info("Running Preprocessing...");
    cv::Mat enhanced = preprocessor.process(image);

    log_info("Running Localization...");
    DefectResult result = localizer.localized(enhanced);

    log_info("Extracting Features (" + std::to_string(result.contours.size()) + " defects)...");
    std::vector<DefectFeatures> features = extractor.extract_all(enhanced, result.contours);

    std::vector<ClassifyResult> classifications;
    if (classifier && !result.bboxes.empty()) {
      log_info("Classifying ROIs...");
      cv::Mat gray;
      if (enhanced.channels() == 3) {
        cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);
      } else {
        gray = enhanced;
      }

      for (auto &bbox : result.bboxes) {
        int x = std::max(0, bbox.x);
        int y = std::max(0, bbox.y);
        int w = std::min(bbox.width, gray.cols - x);
        int h = std::min(bbox.height, gray.rows - y);
        cv::Mat roi = gray(cv::Rect(x, y, w, h)).clone();
        classifications.push_back(classifier->classify(roi));
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double total_ms = elapsed.count();

    if (json_output) {
      std::ostringstream json;
      json << "{\n";
      json << "  \"image\": \"" << json_escape(image_path) << "\",\n";
      json << "  \"resolution\": [" << image.cols << ", " << image.rows << "],\n";
      json << "  \"channels\": " << image.channels() << ",\n";
      json << "  \"processing_ms\": " << total_ms << ",\n";
      json << "  \"model\": \"" << json_escape(model_path) << "\",\n";
      json << "  \"defect_count\": " << result.contours.size() << ",\n";
      json << "  \"defects\": [\n";
      for (size_t i = 0; i < result.contours.size(); ++i) {
        if (i > 0) json << ",\n";
        json << "    {\n";
        json << "      \"id\": " << i << ",\n";
        json << "      \"bbox\": [" << result.bboxes[i].x << ", "
             << result.bboxes[i].y << ", "
             << result.bboxes[i].width << ", "
             << result.bboxes[i].height << "],\n";
        json << "      \"features\": " << features_to_json(features[i]) << ",\n";
        if (i < classifications.size()) {
          json << "      \"classification\": {\n";
          json << "        \"label\": \"" << classifications[i].label_name << "\",\n";
          json << "        \"label_id\": " << classifications[i].label << ",\n";
          json << "        \"confidence\": " << classifications[i].confidence << "\n";
          json << "      }\n";
        } else {
          json << "      \"classification\": null\n";
        }
        json << "    }";
      }
      if (!result.contours.empty()) json << "\n";
      json << "  ]\n";
      json << "}\n";
      std::cout << json.str();
    } else {
      std::cout << "Processing Time: " << total_ms << " ms" << std::endl;
      std::cout << "Defects Found: " << result.contours.size() << std::endl;

      cv::Mat output = image.clone();
      for (size_t i = 0; i < result.bboxes.size(); ++i) {
        cv::rectangle(output, result.bboxes[i], cv::Scalar(0, 0, 255), 2);
        std::string label = "#" + std::to_string(i);
        if (i < classifications.size()) {
          label += " " + classifications[i].label_name +
                   " (" + std::to_string(static_cast<int>(classifications[i].confidence * 100)) + "%)";
        }
        cv::putText(output, label,
                    cv::Point(result.bboxes[i].x, result.bboxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }

      std::string out_filename = "cpp_result.png";
      cv::imwrite(out_filename, output);
      log_info("Result saved to " + out_filename);
      log_info("Pipeline finished successfully.");
    }

    return 0;

  } catch (const cv::Exception &e) {
    log_error("OpenCV Exception: " + std::string(e.what()));
    return -2;
  } catch (const std::invalid_argument &e) {
    log_error("Invalid Input: " + std::string(e.what()));
    return -5;
  } catch (const std::exception &e) {
    log_error("Runtime Exception: " + std::string(e.what()));
    return -3;
  } catch (...) {
    log_error("Unknown Exception occurred");
    return -4;
  }
}
