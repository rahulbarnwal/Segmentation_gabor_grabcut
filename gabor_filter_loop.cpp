//This implemenets gabor filter over a list of images and save the output

//Created by Rahul Barnwal.


#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include"DetectBackground.h"
#include <iostream>
#include<fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <math.h>

using namespace std;
using namespace cv;


///////////////////////////////////////Thinning part /////////////////////////////
void thinningGuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1); 

    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1); 
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = ( (!p2) & (p3 | p4)) + ( (!p4) & (p5 | p6)) +
                     ( (!p6) & (p7 | p8)) + ( (!p8) & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | (!p9) ) & p8) : ((p2 | p3 | (!p5) ) & p4);

            if ( (C == 1) && ((N >= 2 && N <= 3) &  ((m == 0))))
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinningGuoHall(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}

///////////////////////////////////////////////////////




template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

//sigma – Standard deviation of the gaussian envelope.
//theta – Orientation of the normal to the parallel stripes of a Gabor function.
//lambd – Wavelength of the sinusoidal factor.
//gamma – Spatial aspect ratio.
//psi – Phase offset.




void gaborFilter(Mat& greyIm,Mat& final_max)
{
    Mat dest,greyIm_f;
    greyIm.convertTo(greyIm_f,CV_32F);

    int i,j,  kernel_size = 5;
    double sig = 5, gm = 0.04; //double ps = CV_PI/2;
    double ps = CV_PI/2;//PI/2 works better 
    
    std::vector<double> th_arr(6); //theta_array
    for(i=0;i<6;i++)th_arr[i]=30*i;
    
    int rows=greyIm.rows;
    int cols=greyIm.cols;
    
    double lmbd_min=4/sqrt(2);
    double lmbd_max=hypot (rows, cols);
    
    int n=floor(log2(lmbd_max/lmbd_min));
    
    vector<Mat> gaborMat_arr;
    
    std::vector<double> lmbd_arr(n);
    for(i=0;i<n-1;i++)
    {
        lmbd_arr[i]=lmbd_min* pow(2,i); 
    }
    
    for(i=0;i<th_arr.size();i++)
    {
        for(j=0;j<lmbd_arr.size();j++)
        {
            
            cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th_arr[i], lmbd_arr[j], gm, ps);
            cv::filter2D(greyIm_f, dest, CV_32F, kernel);
            gaborMat_arr.push_back(dest);

            //cerr << dest(Rect(30,30,10,10)) << endl; // peek into the data

            cv::Mat Lkernel(kernel_size*10, kernel_size*10, CV_32F);
            cv::resize(kernel, Lkernel, Lkernel.size());
            Lkernel /= 2.;
            Lkernel += 0.5;
            
//            char windowName[50];
//            sprintf(windowName, "Theta= %f and lmda %f", th_arr[i], lmbd_arr[j]);
//            
//            cv::imshow("Kernel", Lkernel);
            
            Mat res;
            dest.convertTo(res,CV_8U,1.0/255.0);     // move to proper[0..255] range to show it

            //cv::imshow(windowName, res);
            
            final_max= max(res,final_max);
            //waitKey();
        }
        
        
        
        
    }
    
//    imshow("Final_image",final_max);
//        waitKey(0);
    
    
    
}

void Contour_superimpose(Mat& binary_img, Mat& image, Mat& super_imposed)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    super_imposed=image;
//    findContours(binary_img,contour,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
    int i;
    RNG rng(12345);
    findContours( binary_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    /// Draw contours
   
    for( i = 0; i< contours.size(); i++ )
    {
//        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        Scalar color = Scalar( 255,0, 0 );
//        drawContours( super_imposed, contours, i, color, -1, 8, hierarchy, 0, Point() );
        drawContours( super_imposed, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }    
    
//    imshow("binary_image",binary_img);
//    imshow("super_imposed",super_imposed);
//    waitKey(0);
    
}



int main()
{   
    
//    ofstream myfile;
//    myfile.open ("/Users/mi0307/grabcut-build/otsuthres_list_after14th.csv");

    ifstream ifsInput("/Users/mi0307/grabcut/Bottomwear_NewStudy.txt");
    string sFileName;  
    Mat final_max;
    while(!ifsInput.eof()) 
        {

             //cout<<"AAAA";
            
            if(ifsInput.eof())break;
            getline(ifsInput, sFileName);
            //Mat image = cv::imread("/Users/mi0307/Downloads/Sizing-Warehouse_cropped/2017-05-11/Trousers/949983/1.bmp",1);
            //
        
            //sFileName="/Users/mi0307/Downloads/Bottomwear_NewStudy_Combined/100305364016.jpg";
            sFileName="/Users/mi0307/Documents/100279481901.jpg";
        
            cout<<sFileName<<endl;
            
            Mat image = cv::imread(sFileName,1);
            medianBlur(image,image,3);
            if( image.empty() )
            {
                cout << "\n Durn, couldn't read image file "<< endl;
                return 1;
            }
            
            Size size((image.size().width)/2,(image.size().height)/2);
            cv::resize(image,image,size);

        
        
            
            Mat greyimage;
            cv::cvtColor(image,greyimage,cv::COLOR_BGR2GRAY);
            
//            imshow("Gray_image",greyimage);
//            waitKey(0);
        
            final_max=Mat::zeros(greyimage.size(),CV_8UC1);
            gaborFilter(greyimage,final_max);
            
        
//            imshow("Gabor_output",final_max);
//            waitKey(0);
        
           // imwrite("/Users/mi0307/Desktop/paper_images/gabor_response.jpg",final_max);
             char c='/';
            vector<string> words=split(sFileName,c);
            vector<string>::iterator ptr;
//            string s="/Users/mi0307/Detectron_mask_folder/binary_mask";
            string s="/Users/mi0307/Documents/";


            vector<string> cloth_detail(words.end()-4,words.end());
            string cloth_detail_joined;
            for(ptr=cloth_detail.begin();ptr!=cloth_detail.end();ptr++)
            {
                cloth_detail_joined=cloth_detail_joined + "_"+*ptr;
            }

            cout<<s+"/"+cloth_detail_joined<<endl;
        
        //applying median filter
            medianBlur(final_max,final_max,3);
//             imshow("on doing median",final_max);
//            waitKey(0);
//        imwrite(s+"/"+cloth_detail_joined, final_max);
//        continue;
//        
        //Code to do thresholding in gabor
            double Min, Max;
            cv::minMaxLoc(final_max, &Min, &Max);


            if (Min!=Max)
            { 
                final_max -= Min;
                final_max.convertTo(final_max,CV_8U,255.0/(Max-Min));
            }


            //cout<<"Source_depth"<<matFG.depth()<<endl;
            double thres_val=cv::threshold(final_max, final_max, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);//otsu threshold
             imshow("on doing threshold",final_max);
            waitKey(0);
            imwrite(s+"/"+"binary_"+cloth_detail_joined, final_max);
//            
//            myfile <<cloth_detail_joined<<","<<thres_val<<"\n";
        
        //code to do thinning
            
//            thinningGuoHall(final_max);
//            imshow("on doing thinning",final_max);
//            waitKey(0);
            
            //imwrite(s+"/"+"binary_"+cloth_detail_joined, final_max);
        
        
            Mat super_imposed=Mat::zeros( image.size(), CV_8UC3 );
            Contour_superimpose(final_max,image,super_imposed);
            
            imshow("Super_imposed_image",super_imposed);
            waitKey(0);
        
        

           imwrite(s+"/_su"+cloth_detail_joined, super_imposed);
            break;
            
        
        }
    return 0;
}




/***GUI_version**  
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

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

void Process(int , void *)
{
    double sig = pos_sigma;
    double lm = 0.5+pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;
    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
    cv::filter2D(src_f, dest, CV_32F, kernel);
    cv::imshow("Process window", dest);
    cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
    cv::resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    cv::imshow("Kernel", Lkernel);
    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    cv::imshow("Mag", mag);
}

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("/Users/mi0307/Downloads/Sizing-Warehouse_cropped/2017-05-09/Shirts/1595113/1.bmp",1);
    cv::imshow("Src", image);
    cv::Mat src;
    cv::cvtColor(image, src, CV_BGR2GRAY);
    src.convertTo(src_f, CV_32F, 1.0/255, 0);
    if (!kernel_size%2)
    {
        kernel_size+=1;
    }
    cv::namedWindow("Process window", 1);
    cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernel_size, Process);
    cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
    cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
    cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    Process(0,0);
    cv::waitKey(0);
    return 0;
}
*/