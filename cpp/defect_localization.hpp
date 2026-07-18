#ifndef WAFERDEFECTX_DEFECT_LOCALIZATION_HPP
#define WAFERDEFECTX_DEFECT_LOCALIZATION_HPP

#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif
#include <vector>

struct DefectResult {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Rect> bboxes;
  cv::Mat mask;
  cv::Mat edges;
};

class DefectLocalizer {
public:
  DefectLocalizer() = default;

  DefectResult localized(const cv::Mat &preprocessed_img);
};

#endif // WAFERDEFECTX_DEFECT_LOCALIZATION_HPP
