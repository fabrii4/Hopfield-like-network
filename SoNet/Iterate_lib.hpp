#include<vector>

#ifndef ITERATELIB_H
#define ITERATELIB_H

//mean pixel value
float mean(std::vector<std::vector<std::vector<float > > >& w, int n, int i, int j, int b);

//iterate the network backward: s_i=Sign(w_ij*s_j)
void Iterate_B(int out [][Ny], std::vector<std::vector<std::vector<float > > >& w, int a, int b, int sens, float err[]);

//iterate the network forward small patterns (convolution): s_i=Sign(w_ij*s_j)
void Iterate_C(int spin [][Ny], int out [][Ny], std::vector<std::vector<float > >& w, std::vector<std::vector<int> >& indx, std::vector<std::vector<int> >& indy, int sens, float sens1, float err[]);

//iterate the network forward small patterns (convolution and pool): s_i=Sign(w_ij*s_j)
void Iterate_CP(int spin [][Ny], int out [][Ny], std::vector<std::vector<float > >& w, std::vector<std::vector<int> >& indx, std::vector<std::vector<int> >& indy, std::vector<std::vector<float > >& w_s, std::vector<std::vector<int> >& indx_s, std::vector<std::vector<int> >& indy_s, int sens, float sens1, float err[], std::vector<int>& scale);



#endif
