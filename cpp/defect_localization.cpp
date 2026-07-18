#include "defect_localization.hpp"

#include <stdexcept>

DefectResult DefectLocalizer::localized(const cv::Mat &preprocessed_img) {
  if (preprocessed_img.empty()) {
    throw std::invalid_argument("DefectLocalizer received empty input image");
  }
  CV_Assert(preprocessed_img.channels() == 1);

  DefectResult result;
  cv::Mat edges, closed;

  cv::Canny(preprocessed_img, edges, 50, 150);
  result.edges = edges.clone();

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  result.mask = cv::Mat::zeros(preprocessed_img.size(), CV_8UC1);

  for (const auto &cnt : contours) {
    double area = cv::contourArea(cnt);
    if (area > 10) {
      result.contours.push_back(cnt);
      result.bboxes.push_back(cv::boundingRect(cnt));
    }
  }

  if (!result.contours.empty()) {
    cv::drawContours(result.mask, result.contours, -1, cv::Scalar(255),
                     cv::FILLED);
  }

  return result;
}
