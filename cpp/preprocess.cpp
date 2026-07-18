#include "preprocess.hpp"

#include <stdexcept>

Preprocessor::Preprocessor(int blur_ksize, int median_ksize)
    : blur_ksize_(blur_ksize), median_ksize_(median_ksize) {
  CV_Assert(blur_ksize_ % 2 == 1);
  CV_Assert(median_ksize_ % 2 == 1);
  clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
}

cv::Mat Preprocessor::process(const cv::Mat &input) {
  if (input.empty()) {
    throw std::invalid_argument("Preprocessor received empty input image");
  }

  cv::Mat gray, blurred, median, enhanced;

  if (input.channels() == 3) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  cv::GaussianBlur(gray, blurred, cv::Size(blur_ksize_, blur_ksize_), 0);
  cv::medianBlur(blurred, median, median_ksize_);
  clahe_->apply(median, enhanced);

  return enhanced;
}

cv::Mat Preprocessor::get_thresholded(const cv::Mat &enhanced) {
  if (enhanced.empty()) {
    return cv::Mat();
  }

  cv::Mat thresh;
  cv::adaptiveThreshold(enhanced, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV, 11, 2);
  return thresh;
}
