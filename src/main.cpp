#include <iostream>
#include "etoeNet.h"
#include "filesystem.h"
// #include <opencv4/opencv2/opencv.hpp>
#include "opencv2/opencv.hpp"

int main(int argc,char**argv){
    etoeNet etoe_net;

    std::string onnx_file_path = "/home/kanghyun/JoyAI_FINAL/src/model-60.simplified.onnx";

    //todo : onnx file -> tensorrt reset
    etoe_net.loadOnnxFile(onnx_file_path);
    //todo : inference image list up
    std::string data_dir = "/home/kanghyun/data1";
    std::vector<std::string> img_extensions = {"jpg","png"};
    std::vector<std::string> file_list;
    listDir(data_dir,file_list, FILE_REGULAR);

    std::vector<std::string> img_list;
    for(auto&file_path : file_list)
    {
        if(fileHasExtension(file_path,img_extensions))
        {
            img_list.push_back(file_path);
        }
    }

    int i = 1;
    int cnt = 1;
    //todo : inference loop
    for(auto&img_path:img_list)
    {
        cv::Mat img_mat = cv::imread(img_path);
        etoe_net.runInference(img_mat);
        std::cout<<"frame : " << cnt <<std::endl;
        cnt++;

        cv::imshow("img",img_mat);
        cv::waitKey(33);
    }
}