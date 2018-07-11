/***GUI_version**  */


/***
This is basic gui version of gabor filter for proper visualisation how parameters affects the segmentation
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;



void Contour_superimpose(Mat& binary_img, Mat& image, Mat& super_imposed)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    super_imposed=image.clone();
//    findContours(binary_img,contour,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
    int i;
    RNG rng(12345);
    findContours( binary_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    /// Draw contours
   
    for( i = 0; i< contours.size(); i++ )
    {
//        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        Scalar color = Scalar( 255,0, 0 );
        drawContours( super_imposed, contours, i, color, 5, 8, hierarchy, 0, Point() );
    }    
    
//    imshow("binary_image",binary_img);
//    imshow("super_imposed",super_imposed);
//    waitKey(0);
    
}

cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{
    int hks = (ks-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(ks-1);
    double lmbd = lm;
    double sigma = sig/ks;
    double x_theta;
    double y_theta;
    cv::Mat kernel(ks,ks, CV_32F);
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

int kernel_size=21;
int pos_sigma= 5;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 90;
cv::Mat src_f;
cv::Mat dest;
cv::Mat image;
Mat super_imposed;
void Process(int , void *)
{
    double sig = pos_sigma;
    double lm = 0.5+pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;
    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
    cv::filter2D(src_f, dest, CV_32F, kernel);
    cv::imshow("Src", image);
    cv::imshow("Process window", dest);
    cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
    cv::resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    cv::imshow("Kernel", Lkernel);
    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    cv::imshow("Mag", mag);
    
    
//    Mat value_window=(image.size(), CV_8UC3);
    Mat dummy_img=image.clone();
    char text1[100];
    char text2[100];
    char text3[100];
    char text4[100];
    sprintf(text1," sigma= %f ",sig);
    sprintf(text2," theta= %f ",th);
    sprintf(text3," lambda= %f ",lm);
    sprintf(text4," psi= %f ", ps);
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 1;
    int thickness = 3;  
    cv::Point textOrg1(10, 130);
    cv::Point textOrg2(30, 50);
    cv::Point textOrg3(50, 200);
    cv::Point textOrg4(70, 300);
    cv::putText(dummy_img, text1, textOrg1, fontFace, fontScale, Scalar::all(255), thickness,8);
    cv::putText(dummy_img, text2, textOrg2, fontFace, fontScale, Scalar::all(255), thickness,8);
    cv::putText(dummy_img, text3, textOrg3, fontFace, fontScale, Scalar::all(255), thickness,8);
    cv::putText(dummy_img, text4, textOrg4, fontFace, fontScale, Scalar::all(255), thickness,8);
    imshow("value",dummy_img);
    
    double Min, Max;
    cv::minMaxLoc(dest, &Min, &Max);


    if (Min!=Max)
    { 
    dest -= Min;
    dest.convertTo(dest,CV_8U,255.0/(Max-Min));
    }


    //cout<<"Source_depth"<<matFG.depth()<<endl;
    double thres_val=cv::threshold(dest, dest, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);//otsu threshold
    
    super_imposed=Mat::zeros( image.size(), CV_8UC3 );
    Contour_superimpose(dest,image,super_imposed);
            
    imshow("Super_imposed_image",super_imposed);
    
    //Code to do thresholding in gabor
    
 }

int main(int argc, char** argv)
{
    image = cv::imread("/Users/mi0307/Downloads/sizing_qc_after14th/14052018/Day shift/100299470421.jpg",1);
    
    Size size((image.size().width)/2,(image.size().height)/2);
    Mat dst;//dst image
    cv::resize(image,image,size);
    
    
    cv::Mat src;
    cv::cvtColor(image, src, CV_BGR2GRAY);
    medianBlur(src,src,3);
    src.convertTo(src_f, CV_32F, 1.0/255, 0);
    if (!kernel_size%2)
    {
        kernel_size+=1;
    }
    cv::namedWindow("Process window", 0);
//    resizeWindow("Display frame", image.size().width, image.size().height);
//    char TrackbarName[50];
// 
//    sprintf( TrackbarName, "%d x %d", pos_sigma, kernel_size);
//    cv::createTrackbar(TrackbarName, "Process window", &pos_sigma, kernel_size, Process);
    cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernel_size, Process);
    cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
    cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
    cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    Process(0,0);
    
    
    
    cv::waitKey(0);
    return 0;
}
