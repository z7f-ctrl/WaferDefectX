#ifndef WAFERDEFECTX_PREPROCESS_HPP
#define WAFERDEFECTX_PREPROCESS_HPP

#include <opencv2/opencv.hpp>

class Preprocessor {
public:
  explicit Preprocessor(int blur_ksize = 5, int median_ksize = 5);

  cv::Mat process(const cv::Mat &input);
  cv::Mat get_thresholded(const cv::Mat &enhanced);

private:
  int blur_ksize_;
  int median_ksize_;
  cv::Ptr<cv::CLAHE> clahe_;
};

#endif // WAFERDEFECTX_PREPROCESS_HPP
