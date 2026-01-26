#include <opencv2/opencv.hpp>
#include <vector>

class Preprocessor {
public:
    Preprocessor(int blur_ksize = 5, int median_ksize = 5) 
        : blur_ksize(blur_ksize), median_ksize(median_ksize) {
        // limit=2.0, tile=8x8
        clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    }

    cv::Mat process(const cv::Mat& input) {
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
    
    cv::Mat get_thresholded(const cv::Mat& enhanced) {
        cv::Mat thresh;
        cv::adaptiveThreshold(enhanced, thresh, 255, 
            cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
        return thresh;
    }

private:
    int blur_ksize;
    int median_ksize;
    cv::Ptr<cv::CLAHE> clahe;
};
