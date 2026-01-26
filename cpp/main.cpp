#include "defect_localization.cpp"
#include "preprocess.cpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: ./WaferDefectX_Run <image_path>" << std::endl;
    return -1;
  }

  std::string image_path = argv[1];
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  // Initialize modules
  Preprocessor preprocessor;
  DefectLocalizer localizer;

  // Start Timer
  auto start = std::chrono::high_resolution_clock::now();

  // 1. Preprocess
  cv::Mat enhanced = preprocessor.process(image);

  // 2. Localize
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

  cv::imwrite("cpp_result.png", output);

  return 0;
}
