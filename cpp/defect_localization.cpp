#include <opencv2/opencv.hpp>
#include <vector>

struct DefectResult {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> bboxes;
    cv::Mat mask;
    cv::Mat edges;
};

class DefectLocalizer {
public:
    DefectLocalizer() {}

    DefectResult localized(const cv::Mat& preprocessed_img) {
        DefectResult result;
        cv::Mat edges, closed;
        
        // 1. Canny
        cv::Canny(preprocessed_img, edges, 50, 150);
        result.edges = edges.clone();

        // 2. Morphology Close
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);

        // 3. Find Contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        result.mask = cv::Mat::zeros(preprocessed_img.size(), CV_8UC1);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area > 10) {
                result.contours.push_back(cnt);
                result.bboxes.push_back(cv::boundingRect(cnt));
            }
        }
        
        // Draw Mask
        cv::drawContours(result.mask, result.contours, -1, cv::Scalar(255), cv::FILLED);

        return result;
    }
};
