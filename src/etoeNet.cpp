#include "etoeNet.h"

etoeNet::etoeNet() : tensorNet()
{

}

etoeNet::~etoeNet()
{
}

void etoeNet::loadOnnxFile(const std::string &onnx_file_path)
{   
    
    std::vector<std::string> input_blobs = {"input"};
    std::vector<std::string> output_blobs = {"output"};
    LoadNetwork(NULL, onnx_file_path.c_str(), NULL, input_blobs, output_blobs,1, TYPE_FP32);   


    m_img_crpped_rgb_f_mat = cv::Mat(cv::Size(320,70),CV_32FC3, mInputs[0].CPU);
    //m_img_crpped_rgb_f_mat2 = cv::Mat(cv::Size(320,70),CV_32FC3, mInputs[1].CPU); 
} 



void etoeNet::runInference(const cv::Mat &img_mat)
{
    //--------------------------------------------
    // Preprocess
    cv::Mat img_crpped_mat = img_mat(cv::Rect(0,65,320,70));
    //cv::Mat img_crpped_mat2 = img_mat(cv::Rect(0,65,320,70));

    cv::Mat img_crpped_rgb_mat;
   //cv::Mat img_crpped_rgb_mat2;
    cv::cvtColor(img_crpped_mat, img_crpped_rgb_mat, cv::COLOR_BGR2RGB);
   //cv::cvtColor(img_crpped_mat2, img_crpped_rgb_mat2, cv::COLOR_BGR2RGB);


    img_crpped_rgb_mat.convertTo(m_img_crpped_rgb_f_mat, CV_32FC3,1.0/255.0);
    //img_crpped_rgb_mat2.convertTo(m_img_crpped_rgb_f_mat2, CV_32FC3,1.0/255.0);

    //----------------------------------------------
    ProcessNetwork(true);

    std::cout <<"run"<<std::endl;

    //todo :inference resul
    float network_output_angle = *(mOutputs[0].CPU);

    float steering_angle;
    float network_output_velocity =-*(mOutputs[0].CPU+1);
    if(network_output_angle<-0.75)
    {
        steering_angle = -1.00;
    }
    else if (network_output_angle< -0.25)
    {
        steering_angle = -0.5;
    }
     else if (network_output_angle< 0.25)
    {
        steering_angle = 0;
    }
     else if (network_output_angle< 0.75)
    {
        steering_angle = 0.5;
    }
     else
    {
        steering_angle = 1.00;
    }

    steering_angle *=20.0;

    std::cout<<"network_output_angle  : " <<network_output_angle << std::endl;
    std::cout<<"actual steering angle : " <<steering_angle <<std::endl;
    std::cout<<"network_output_velocity : " << network_output_velocity <<std::endl<<std::endl;
}
 
