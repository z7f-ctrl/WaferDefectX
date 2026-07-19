#include "onnx_classifier.hpp"
#include <stdexcept>
#include <algorithm>

static const OrtApi *g_ort = nullptr;

static void check_status(OrtStatus *s) {
  if (s) {
    const char *msg = g_ort->GetErrorMessage(s);
    std::string err(msg);
    g_ort->ReleaseStatus(s);
    throw std::runtime_error(err);
  }
}

OnnxClassifier::OnnxClassifier(const std::string &model_path)
    : env_(nullptr), session_(nullptr), allocator_(nullptr),
      input_width_(64), input_height_(64) {

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "WaferDefectX", &env_));

  OrtSessionOptions *opts = nullptr;
  check_status(g_ort->CreateSessionOptions(&opts));
  check_status(g_ort->SetIntraOpNumThreads(opts, 1));

  check_status(g_ort->CreateSession(env_, model_path.c_str(), opts, &session_));
  g_ort->ReleaseSessionOptions(opts);

  check_status(g_ort->GetAllocatorWithDefaultOptions(&allocator_));

  size_t num_inputs = 0, num_outputs = 0;
  check_status(g_ort->SessionGetInputCount(session_, &num_inputs));
  check_status(g_ort->SessionGetOutputCount(session_, &num_outputs));

  input_names_.resize(num_inputs);
  output_names_.resize(num_outputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    char *name = nullptr;
    check_status(g_ort->SessionGetInputName(session_, i, allocator_, &name));
    input_names_[i] = std::string(name);
    (void)g_ort->AllocatorFree(allocator_, name);
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    char *name = nullptr;
    check_status(g_ort->SessionGetOutputName(session_, i, allocator_, &name));
    output_names_[i] = std::string(name);
    (void)g_ort->AllocatorFree(allocator_, name);
  }

  input_names_c_.clear();
  for (auto &n : input_names_) input_names_c_.push_back(n.c_str());
  output_names_c_.clear();
  for (auto &n : output_names_) output_names_c_.push_back(n.c_str());
}

OnnxClassifier::~OnnxClassifier() {
  if (session_) g_ort->ReleaseSession(session_);
  if (env_) g_ort->ReleaseEnv(env_);
}

ClassifyResult OnnxClassifier::classify(const cv::Mat &roi) {
  cv::Mat gray, resized;
  if (roi.channels() == 3) {
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = roi;
  }
  cv::resize(gray, resized, cv::Size(input_width_, input_height_));
  resized.convertTo(resized, CV_32FC1, 1.0 / 255.0);

  std::vector<float> input_tensor_values(
      (float *)resized.data,
      (float *)resized.data + input_width_ * input_height_);

  int64_t input_shape[] = {1, 1, input_height_, input_width_};
  OrtMemoryInfo *mem_info = nullptr;
  check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

  OrtValue *input_tensor = nullptr;
  check_status(g_ort->CreateTensorWithDataAsOrtValue(
      mem_info,
      input_tensor_values.data(),
      input_tensor_values.size() * sizeof(float),
      input_shape, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &input_tensor));
  g_ort->ReleaseMemoryInfo(mem_info);

  OrtValue *output_tensor = nullptr;
  check_status(g_ort->Run(
      session_, nullptr,
      input_names_c_.data(), &input_tensor, 1,
      output_names_c_.data(), 1,
      &output_tensor));

  float *output_data = nullptr;
  check_status(g_ort->GetTensorMutableData(output_tensor, (void **)&output_data));

  static const char *labels[] = {"good", "particle", "scratch"};
  int num_classes = 3;

  int pred = 0;
  float max_val = output_data[0];
  for (int i = 1; i < num_classes; ++i) {
    if (output_data[i] > max_val) {
      max_val = output_data[i];
      pred = i;
    }
  }

  float sum = 0;
  for (int i = 0; i < num_classes; ++i) sum += std::exp(output_data[i]);
  float confidence = std::exp(output_data[pred]) / sum;

  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);

  return {pred, confidence, labels[pred]};
}

std::vector<ClassifyResult> OnnxClassifier::classify_batch(
    const std::vector<cv::Mat> &rois) {
  std::vector<ClassifyResult> results;
  results.reserve(rois.size());
  for (auto &roi : rois) {
    results.push_back(classify(roi));
  }
  return results;
}
