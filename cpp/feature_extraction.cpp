#include "feature_extraction.hpp"

#include <cmath>

DefectFeatures FeatureExtractor::extract(const cv::Mat &gray,
                                         const std::vector<cv::Point> &contour) {
  DefectFeatures f{};

  double area = cv::contourArea(contour);
  double perimeter = cv::arcLength(contour, true);

  double circularity = 0.0;
  if (perimeter > 0) {
    circularity = (4.0 * CV_PI * area) / (perimeter * perimeter);
  }

  cv::Rect bbox = cv::boundingRect(contour);
  double aspect_ratio = (bbox.height > 0) ? static_cast<double>(bbox.width) / bbox.height : 0.0;
  double rectangularity = (bbox.width * bbox.height > 0) ? area / (bbox.width * bbox.height) : 0.0;

  cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
  std::vector<std::vector<cv::Point>> cnts = {contour};
  cv::drawContours(mask, cnts, -1, cv::Scalar(255), cv::FILLED);

  cv::Scalar mean_val, std_val;
  cv::meanStdDev(gray, mean_val, std_val, mask);

  f.values[0] = area;
  f.values[1] = perimeter;
  f.values[2] = aspect_ratio;
  f.values[3] = rectangularity;
  f.values[4] = circularity;
  f.values[5] = mean_val[0];
  f.values[6] = std_val[0];

  return f;
}

std::vector<DefectFeatures>
FeatureExtractor::extract_all(const cv::Mat &gray,
                              const std::vector<std::vector<cv::Point>> &contours) {
  std::vector<DefectFeatures> results;
  results.reserve(contours.size());
  for (const auto &cnt : contours) {
    results.push_back(extract(gray, cnt));
  }
  return results;
}
