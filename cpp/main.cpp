#include "defect_localization.cpp"
#include "preprocess.cpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

// Simple structured logging helper
void log_info(const std::string &msg) {
  std::cout << "[INFO] " << msg << std::endl;
}

void log_error(const std::string &msg) {
  std::cerr << "[ERROR] " << msg << std::endl;
}

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      log_error("Usage: ./WaferDefectX_Run <image_path>");
      return -1;
    }

    std::string image_path = argv[1];
    log_info("Starting WaferDefectX C++ Pipeline");
    log_info("Loading image: " + image_path);

    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
      throw std::runtime_error("Could not open or find the image at " +
                               image_path);
    }

    log_info("Image loaded successfully. Resolution: " +
             std::to_string(image.cols) + "x" + std::to_string(image.rows));

    // Initialize modules
    Preprocessor preprocessor;
    DefectLocalizer localizer;

    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Preprocess
    log_info("Running Preprocessing (Blur, CLAHE)...");
    cv::Mat enhanced = preprocessor.process(image);

    // 2. Localize
    log_info("Running Localization (Canny, Morphology, Contours)...");
    DefectResult result = localizer.localized(enhanced);

    // End Timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> float_ms = end - start;

    std::cout << "Processing Time: " << float_ms.count() << " ms" << std::endl;
    std::cout << "Defects Found: " << result.contours.size() << std::endl;

    // Save result for verification
    cv::Mat output = image.clone();
    for (size_t i = 0; i < result.contours.size(); i++) {
      cv::rectangle(output, result.bboxes[i], cv::Scalar(0, 0, 255), 2);
    }

    std::string out_filename = "cpp_result.png";
    cv::imwrite(out_filename, output);
    log_info("Result saved to " + out_filename);
    log_info("Pipeline finished successfully.");

    return 0;

  } catch (const cv::Exception &e) {
    log_error("OpenCV Exception: " + std::string(e.what()));
    return -2;
  } catch (const std::exception &e) {
    log_error("Runtime Exception: " + std::string(e.what()));
    return -3;
  } catch (...) {
    log_error("Unknown Exception occurred");
    return -4;
  }
}
