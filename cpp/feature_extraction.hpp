#ifndef WAFERDEFECTX_FEATURE_EXTRACTION_HPP
#define WAFERDEFECTX_FEATURE_EXTRACTION_HPP

#include <opencv2/opencv.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif
#include <array>
#include <vector>

constexpr int FEATURE_DIM = 7;

struct DefectFeatures {
  std::array<double, FEATURE_DIM> values;
};

class FeatureExtractor {
public:
  DefectFeatures extract(const cv::Mat &gray, const std::vector<cv::Point> &contour);

  std::vector<DefectFeatures> extract_all(const cv::Mat &gray,
                                          const std::vector<std::vector<cv::Point>> &contours);
};

#endif // WAFERDEFECTX_FEATURE_EXTRACTION_HPP
