#ifndef __ETOE_NET__
#define __ETOE_NET__

#include <opencv4/opencv2/opencv.hpp>
#include "tensorNet.h"                                                                                                                         

class etoeNet : tensorNet
{
public:
    etoeNet();
    ~etoeNet();

    void loadOnnxFile(const std::string &onnx_file_path);


    void runInference(const cv::Mat &img_mat);


private:
    //float *m_trt_input_data;
    cv::Mat m_img_crpped_rgb_f_mat;
    cv::Mat m_img_crpped_rgb_f_mat2;
};

#endif