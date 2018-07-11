//This implemenets grabcut over a list of images and save the output

//Created by Rahul Barnwal.



#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include<fstream>
#include <sstream>
#include <vector>
#include <iterator>


using namespace std;
using namespace cv;

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







Mat mask;
static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

void reset()
{
    if( !mask.empty() )
    {   
        mask.setTo(Scalar::all(GC_BGD));
    }
}

void setRectInMask(Rect bounding_rect)
{
    
    CV_Assert( !mask.empty() );
    cout<<"SETLrectinmask"<<endl;
    
    (mask(bounding_rect)).setTo( Scalar(GC_PR_FGD));
}

int main( int argc, char** argv )
{
    ifstream ifsInput("/Users/mi0307/grabcut/file_list.txt");
    string sFileName;   
    while(!ifsInput.eof()) 
    {
   
         //cout<<"AAAA";
        if(ifsInput.eof())break;
        getline(ifsInput, sFileName);
        //Mat image = cv::imread("/Users/mi0307/Downloads/Sizing-Warehouse_cropped/2017-05-11/Trousers/949983/1.bmp",1);
        //cout<<sFileName<<endl;
        
        Mat image = cv::imread(sFileName,1);
        
        if( image.empty() )
        {
            cout << "\n Durn, couldn't read image file "<< endl;
            return 1;
        }

//        cv::Rect boundrect(15,15,image.cols-30,image.rows-30); //UNCOMMENT IT !!!
        
        cv::Rect boundrect(30,30,image.cols-60,image.rows-60);
        
        
        //cv::rectangle(image,boundrect,cv::Scalar(0, 255, 0));
        //imwrite("/Users/mi0307/grabcut-build/bounded.bmp", image);



        Mat res,binMask;
        Mat bgdModel,fgdModel;
        //CV_Assert( !mask.empty() );
        mask.create( image.size(), CV_8UC1);
        reset();

        setRectInMask(boundrect);
        //imwrite("/Users/mi0307/grabcut-build/mask1.bmp", mask);
        
        grabCut( image, mask, boundrect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );



        getBinMask( mask, binMask );
        image.copyTo( res, binMask );

        //imwrite("/Users/mi0307/grabcut-build/res.bmp", res);


        int i;
        for(i=0;i<10;i++)
        {
            res.release();
            grabCut( image, mask, boundrect, bgdModel, fgdModel, 1);
            getBinMask( mask, binMask );
            image.copyTo( res, binMask );

            //imwrite("/Users/mi0307/grabcut-build/res"+ std::to_string(i)+".bmp", res);
        }
        
        
        char c='/';
        vector<string> words=split(sFileName,c);
        vector<string>::iterator ptr;
        string s="/Users/mi0307/grabcut-build/grabcut_segmented_image_smaller_rec";
        
    
        vector<string> cloth_detail(words.end()-4,words.end());
        string cloth_detail_joined;
        for(ptr=cloth_detail.begin();ptr!=cloth_detail.end();ptr++)
        {
            cloth_detail_joined=cloth_detail_joined + "_"+*ptr;
        }
        
        //cout<<"The s="<<s+"/"+cloth_detail_joined<<endl;
        imwrite(s+"/"+cloth_detail_joined, res);
        
        //break;
        
//        mask.setTo(GC_BGD);
//        (mask(boundrect)).setTo( Scalar(GC_PR_FGD));

       /*** grabCut( image, mask, boundrect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );

        imwrite("/Users/mi0307/grabcut-build/mask2.bmp", mask);

        getBinMask( mask, binMask );
        image.copyTo( res, binMask );

        imwrite("/Users/mi0307/grabcut-build/res2.bmp", res);
        // mask.setTo(GC_BGD);
       */
        
    }
    return 0;
}
