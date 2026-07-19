#ifndef WAFERDEFECTX_ONNX_CLASSIFIER_HPP
#define WAFERDEFECTX_ONNX_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_c_api.h>
#include <string>
#include <vector>

struct ClassifyResult {
  int label;
  float confidence;
  std::string label_name;
};

class OnnxClassifier {
public:
  OnnxClassifier(const std::string &model_path);
  ~OnnxClassifier();

  ClassifyResult classify(const cv::Mat &roi);

  std::vector<ClassifyResult> classify_batch(const std::vector<cv::Mat> &rois);

private:
  OrtEnv *env_;
  OrtSession *session_;
  OrtAllocator *allocator_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_names_c_;
  std::vector<const char *> output_names_c_;
  int input_width_;
  int input_height_;
};

#endif // WAFERDEFECTX_ONNX_CLASSIFIER_HPP
