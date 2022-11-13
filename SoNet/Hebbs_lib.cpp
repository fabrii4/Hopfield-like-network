#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<vector>
#include<ctime>
#include<cmath>

#include "Hebbs_lib.hpp"

using namespace cv;
using namespace std;


//Spin multiplication
int sX(int x, int y) 
{
//   return (x ^ y)^255;
   //return exp(-pow(x-y,2)/255)*255;
   if(x-y<10&&x-y>0||x-y>-10&&x-y<0)
      return 255;
   else
      return 0;

}

//convert image to spin configuration
void ImageToSpin(Mat& img, int spin[][Ny])
{
    for(int i=0;i<img.cols;i++)
    {
       for(int j=0;j<img.rows;j++)
       {
          Scalar intens=img.at<uchar>(j,i);
          spin[i][j]=(int)intens[0];
       }
    }
}

//convert spin configuration to image
void SpinToImage(Mat& img, int spin[][Ny])
{
    for(int i=0;i<img.cols;i++)
    {
       for(int j=0;j<img.rows;j++)
       {
          int intens=1*spin[i][j];
          img.at<uchar>(j,i)=intens;
       }
    }
}

//Resize pattern
void Pool(int spin[][Ny], int out[][Ny], int scale)
{
    for(int i=0;i<Nx;i+=scale)
    {
       for(int j=0;j<Ny;j+=scale)
       {
          int smax=0;
          for(int k=0;k<scale;k++)
          {
             for(int l=0;l<scale;l++)
             {
                if(i+k<Nx && j+l<Ny && spin[i+k][j+l]>smax)
                   smax=spin[i+k][j+l];
             }
          }
          out[i/scale][j/scale]=smax;
       }
    }
}

//Learn pattern
void Hebbs(int spin [][Ny], vector<vector<vector<float > > >& w)
{   
    float A=1.;//(Nx*Ny*256.);
    vector<vector<float > > sk;
    for(int k=0;k<Nx;k++)
    {
       vector<float > sl;
       for(int l=0;l<Ny;l++)
       {
          sl.push_back(A*spin[k][l]);
       }
       sk.push_back(sl);
    }
    w.push_back(sk);  
}

//Learn pattern (clustering neurons with similar value)
void Hebbs_clust(int spin [][Ny], vector<vector<vector<vector<float > > > >& w, int sens, float err[])
{   
    float A=1.;//(Nx*Ny*256.);
    vector<vector<vector<float > > > wc;
    vector<vector<float > > sk;
    for(int k=0;k<Nx;k++)
    {
       vector<float > sl;
       for(int l=0;l<Ny;l++)
       {
          if(k-1>=0 && abs(spin[k][l]-spin[k-1][l])<=sens*err[spin[k][l]])
          sl.push_back(A*spin[k][l]);
       }
       sk.push_back(sl);
    }
    wc.push_back(sk);  
}

//Learn segmented pattern
void Hebbs_p(int spin [][Ny], int mask [][Ny], vector<vector<float > >& w, vector<vector<int> >& indx, vector<vector<int> >& indy)
{   
    float A=1.;//(Nx*Ny*256.);
    int imax=0,jmax=0;
    vector<float> wt;
    vector<int> xt,yt;
    for(int i=0;i<Nx;i++)
    {
       for(int j=0;j<Ny;j++)
       {
          if(mask[i][j]>100)
          {
             wt.push_back(A*spin[i][j]);
             xt.push_back(i);
             yt.push_back(j);
             imax+=i;
             jmax+=j;
          }
       }
    }
    imax=imax/(float)xt.size();
    jmax=jmax/(float)yt.size();    
    for(int i=0;i<xt.size();i++)
    {
       xt[i]+=-imax;
       yt[i]+=-jmax;
    }
    //for(n=0;n<w.size();n++)
    //{

    //}
    if(wt.size()>15)
    {
       w.push_back(wt);
       indx.push_back(xt);
       indy.push_back(yt);
    }
}

//Learn segmented pattern and scaled version
void Hebbs_pP(int spin [][Ny], int mask [][Ny], vector<vector<float > >& w, vector<vector<int> >& indx, vector<vector<int> >& indy, vector<vector<float > >& w_s, vector<vector<int> >& indx_s, vector<vector<int> >& indy_s, vector<int>& scale)
{   
    float A=1.;//(Nx*Ny*256.);
    int imax=0,jmax=0; 
    int im=Nx,jm=Ny,iM=0,jM=0;
    vector<float> wt;
    vector<int> xt,yt;
    for(int i=0;i<Nx;i++)
    {
       for(int j=0;j<Ny;j++)
       {
          if(mask[i][j]>100)
          {
             wt.push_back(A*spin[i][j]);
             xt.push_back(i);
             yt.push_back(j);
             imax+=i;
             jmax+=j;
             if(i<im) im=i;
             if(j<jm) jm=j;
             if(i>iM) iM=i;
             if(j>jM) jM=j;
          }
       }
    }
    imax=imax/(float)xt.size();
    jmax=jmax/(float)yt.size();    
    for(int i=0;i<xt.size();i++)
    {
       xt[i]+=-imax;
       yt[i]+=-jmax;
    }
    //for(n=0;n<w.size();n++)
    //{

    //}
    if(wt.size()>36)
    {
       w.push_back(wt);
       indx.push_back(xt);
       indy.push_back(yt);
        
       int s=min(8,min(iM-im,jM-jm)/8);
       int imaxs=0,jmaxs=0; 
       vector<float> wts;
       vector<int> xts,yts;
       int spin_s[Nx][Ny], mask_s[Nx][Ny];
       Pool(spin,spin_s,s);
       Pool(mask,mask_s,s);
       for(int i=0;i<Nx/s;i++)
       {
          for(int j=0;j<Ny/s;j++)
          {
             if(mask_s[i][j]>100)
             {
                wts.push_back(A*spin_s[i][j]);
                xts.push_back(i);
                yts.push_back(j);
                imaxs+=i;
                jmaxs+=j;
             }
          }
       }
       imaxs=imaxs/(float)xts.size();
       jmaxs=jmaxs/(float)yts.size();    
       for(int i=0;i<xts.size();i++)
       {
          xts[i]+=-imaxs;
          yts[i]+=-jmaxs;
       }
       w_s.push_back(wts);
       indx_s.push_back(xts);
       indy_s.push_back(yts);
       scale.push_back(s);

    }
}
