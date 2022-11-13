#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bitset>
#include <iostream>
#include<string>
#include<vector>
#include<ctime>
#include<cmath>

#include <unistd.h>
#include<fstream>

#include "Hebbs_lib.hpp"
#include "Iterate_lib.hpp"

//video size
#define Nx 120
#define Ny 100
//convolution size
#define L 1

float err[256];

using namespace cv;
using namespace std;




int main()
{
    Mat rgb;
    Mat img(Size(Nx,Ny),CV_8U);
    Mat output(Size(Nx,Ny),CV_8U);
    Mat output1(Size(Nx,Ny),CV_8U);
    VideoCapture cam(0);

    //Hebbs network
    int in[Nx][Ny],in1[Nx][Ny], out[Nx][Ny], s_mask[Nx][Ny];
    vector<vector<vector<float > > > w;
    vector<vector<float> > wp, wps; 
    vector<vector<int> > indx,indy,indxs,indys; 
    vector<int> scale;
    int sel=0,segm=0,see=0,sens=5;
    float sens1=0.6;

    for(int i=0;i<256;i++)
    {
       err[i]=(16.29*exp(-0.0356*i)+0.95)*i/100.;
    }


    srand(time(0));
    

    namedWindow("input",WINDOW_NORMAL);
    moveWindow("input",1920+100,100);
    resizeWindow("input", 200,200);
    namedWindow("output",WINDOW_NORMAL);
    moveWindow("output",1920+400,100);
    resizeWindow("output", 200,200);
    namedWindow("output1",WINDOW_NORMAL);
    moveWindow("output1",1920+100,400);
    resizeWindow("output1", 200,200);

    while(true)
    {
        cam>>rgb;
        cvtColor(rgb,rgb,COLOR_RGB2GRAY);
        resize(rgb,img,Size(Nx,Ny));
        //imshow("input",img);

        Mat img1(Size(Nx,Ny),CV_8U,Scalar(0));

        ImageToSpin(img,in);
        ImageToSpin(img,in1);
        Hebbs(in,w);

        if(w.size()>11 && sel==0)
        {
           //Iterate_B(in,w,w.size()-1,10,sens);
           Iterate_B(in,w,w.size()-1,w.size()-2,sens,err);
           int En=0;
           for(int i=0;i<Nx;i++)
           {
              for(int j=0;j<Ny;j++)
              {
                 En+=in[i][j];
              }
           }
           //cout<<En<<endl;
           if(En<20000) w.pop_back();
           else
           {
              Mat mask(Size(Nx,Ny),CV_8U);
              for(int i=0;i<Nx;i++)
              {
                 for(int j=0;j<Ny;j++)
                 {
                    in[i][j]+=(in[i][j]<100)*in1[i][j];
                 }
              }
              SpinToImage(mask,in);
              //threshold(mask, mask, 15, 255, CV_THRESH_BINARY);
              //img.copyTo(img1, mask);
              mask.copyTo(img1);
              if(segm==1)
              {
                 for(int i=0;i<Nx;i++)
                 {
                    for(int j=0;j<Ny;j++)
                    {
                       s_mask[i][j]=in[i][j];
                    }
                 }
              }

              imshow("output",img1);
              //Iterate_D(outD,w,w.size()-1,w.size()-2);
              //SpinToImage(output1,outD[0]);
              //imshow("output1",output1);
           }
        }
        if(w.size()>1000) w.clear();

        if(segm==1)
        {
           //Hebbs_p(in1,s_mask,wp,indx,indy);
           Hebbs_pP(in1,s_mask,wp,indx,indy,wps,indxs,indys,scale);
           cout<<"wp size "<<wp.size()<<endl;
        }
        if(see==1 && wp.size()>0)
        {
           //Iterate_C(in1,out,wp,indx,indy,sens,sens1);
           Iterate_CP(in1,out,wp,indx,indy,wps,indxs,indys,sens,sens1,err,scale);
           SpinToImage(output1,out);
           imshow("output1",output1);
        }


        imshow("input",img);


        char c = waitKey(10);
        if(27 == char(c)) break;
        if(c=='1') sel=(sel+1)%2;    //calculate subtraction mask
        if(c=='2') segm=(segm+1)%2;  //learn mask pattern
        if(c=='3') see=(see+1)%2;    //apply mask patterns
        if(c=='4' && wp.size()>0)                   //forget last pattern
        {
           wp.pop_back();
           indx.pop_back();
           indy.pop_back();
           cout<<"wp size "<<wp.size()<<endl;
        }
        if(c=='5') {sens+=1;cout<<"sens "<<sens<<endl;}
        if(c=='6') {sens+=-1;cout<<"sens "<<sens<<endl;}
        if(c=='7') {sens1+=0.1;cout<<"sens1 "<<sens1<<endl;}
        if(c=='8') {sens1+=-0.1;cout<<"sens1 "<<sens1<<endl;}
        if(c=='9') {w.clear();cout<<"cleared"<<endl;}


    }


    return 0;
}
