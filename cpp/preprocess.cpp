#include <opencv2/opencv.hpp>
#include <vector>

class Preprocessor {
public:
  Preprocessor(int blur_ksize = 5, int median_ksize = 5)
      : blur_ksize(blur_ksize), median_ksize(median_ksize) {
    // Assert valid kernel sizes (must be odd)
    CV_Assert(blur_ksize % 2 == 1);
    CV_Assert(median_ksize % 2 == 1);

    // limit=2.0, tile=8x8
    clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  }

  cv::Mat process(const cv::Mat &input) {
    // Defensive check: Empty input
    if (input.empty()) {
      throw std::invalid_argument("Preprocessor received empty input image");
    }

    cv::Mat gray, blurred, median, enhanced, thresholded;

    // 1. Convert to Gray
    if (input.channels() == 3) {
      cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = input.clone();
    }

    // 2. Gaussian Blur
    cv::GaussianBlur(gray, blurred, cv::Size(blur_ksize, blur_ksize), 0);

    // 3. Median Blur
    cv::medianBlur(blurred, median, median_ksize);

    // 4. CLAHE
    clahe->apply(median, enhanced);

    return enhanced;
  }

  cv::Mat get_thresholded(const cv::Mat &enhanced) {
    if (enhanced.empty())
      return cv::Mat();

    cv::Mat thresh;
    cv::adaptiveThreshold(enhanced, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 11, 2);
    return thresh;
  }

private:
  int blur_ksize;
  int median_ksize;
  cv::Ptr<cv::CLAHE> clahe;
};
