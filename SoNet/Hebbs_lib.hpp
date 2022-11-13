#include <opencv2/opencv.hpp>
#include<vector>


//video size
#define Nx 120
#define Ny 100

#ifndef HEBBSLIB_H
#define HEBBSLIB_H


//Spin multiplication
int sX(int x, int y);

//convert image to spin configuration
//void ImageToSpin(cv::Mat& img, int spin[][Ny]);
void ImageToSpin(cv::Mat& img, int spin [][Ny]);

//convert spin configuration to image
//void SpinToImage(cv::Mat& img, int spin[][Ny]);
void SpinToImage(cv::Mat& img, int spin [][Ny]);

//Resize pattern
void Pool(int spin[][Ny], int out[][Ny], int scale);

//Learn pattern
void Hebbs(int spin [][Ny], std::vector<std::vector<std::vector<float > > >& w);

//Learn pattern (clustering neurons with similar value)
void Hebbs_clust(int spin [][Ny], std::vector<std::vector<std::vector<std::vector<float > > > >& w, int sens, float err[]);

//Learn segmented pattern
void Hebbs_p(int spin [][Ny], int mask [][Ny], std::vector<std::vector<float > >& w, std::vector<std::vector<int> >& indx, std::vector<std::vector<int> >& indy);

//Learn segmented pattern and scaled version
void Hebbs_pP(int spin [][Ny], int mask [][Ny], std::vector<std::vector<float > >& w, std::vector<std::vector<int> >& indx, std::vector<std::vector<int> >& indy, std::vector<std::vector<float > >& w_s, std::vector<std::vector<int> >& indx_s, std::vector<std::vector<int> >& indy_s, std::vector<int>& scale);


#endif
