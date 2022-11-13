#include<vector>
#include<ctime>
#include<cmath>

#include "Hebbs_lib.hpp"
#include "Iterate_lib.hpp"

using namespace std;


//mean pixel value
float mean(vector<vector<vector<float > > >& w, int n, int i, int j, int b)
{
   float wm=0, sw=0;
   for(int m=0;m<n;m++)
   {
      wm+=w[max(b-m,0)][i][j];
   }
   return wm/n;
}

//iterate the network backward: s_i=Sign(w_ij*s_j)
void Iterate_B(int out [][Ny], vector<vector<vector<float > > >& w, int a, int b, int sens, float err[])
{
    for(int i=0;i<Nx;i++)
    {
       for(int j=0;j<Ny;j++)
       {
          //out[i][j]=abs(w[a][i][j]-w[b][i][j]);
          out[i][j]=(abs(w[a][i][j]-w[b][i][j])>sens*err[(int)w[a][i][j]])*255;
          //out[i][j]=(abs(w[a][i][j]-mean(w,50,i,j,b))>sens*err[(int)w[a][i][j]])*255;
          //out[i][j]=(int)(uint8_t)~sX(w[a][i][j],w[b][i][j]);
       }
    }    
}

//iterate the network forward small patterns (convolution): s_i=Sign(w_ij*s_j)
void Iterate_C(int spin [][Ny], int out [][Ny], vector<vector<float > >& w, vector<vector<int> >& indx, vector<vector<int> >& indy, int sens, float sens1, float err[])
{
    int stride=3;
    for(int i=0;i<Nx;i++)
    {
       for(int j=0;j<Ny;j++)
       {
          out[i][j]=spin[i][j]/2;
       }
    }
    for(int n=0;n<w.size();n++)
    {
       int hmax=0,imax=0,jmax=0;
       for(int i=0;i<Nx;i+=stride)
       {
          for(int j=0;j<Ny;j+=stride)
          {
             int h=0;
             for(int k=0;k<indx[n].size();k++)
             {
                if(i+indx[n][k]>0 && i+indx[n][k]<Nx && j+indy[n][k]>0 && j+indy[n][k]<Ny)
                {
                   if(abs(w[n][k]-spin[i+indx[n][k]][j+indy[n][k]])<sens*err[(int)w[n][k]])//25)
                      h+=1;
                   else
                      h+=-1;
                }
             }
             if(h>hmax) {hmax=h;imax=i;jmax=j;}
          }
       }
       if(hmax>sens1*w[n].size()) 
       {
          for(int k=0;k<indx[n].size();k++)
          {
             if(imax+indx[n][k]>0 && imax+indx[n][k]<Nx && jmax+indy[n][k]>0 && jmax+indy[n][k]<Ny)
             {
                //out[i+indx[nmax][k]][j+indy[nmax][k]]=spin[i+indx[nmax][k]][j+indy[nmax][k]];
                if(abs(w[n][k]-spin[imax+indx[n][k]][jmax+indy[n][k]])<sens*err[(int)w[n][k]])
                   out[imax+indx[n][k]][jmax+indy[n][k]]=255;
             }
          }
       }    
    }
}

//iterate the network forward small patterns (convolution and pool): s_i=Sign(w_ij*s_j)
void Iterate_CP(int spin [][Ny], int out [][Ny], vector<vector<float > >& w, vector<vector<int> >& indx, vector<vector<int> >& indy, vector<vector<float > >& w_s, vector<vector<int> >& indx_s, vector<vector<int> >& indy_s, int sens, float sens1, float err[], vector<int>& scale)
{
    int stride=1;
    for(int i=0;i<Nx;i++)
    {
       for(int j=0;j<Ny;j++)
       {
          out[i][j]=spin[i][j]/2;
       }
    }
    for(int n=0;n<w.size();n++)
    {
       int s_p[Nx][Ny];
       Pool(spin,s_p,scale[n]);
       int hmax_s=0,imax_s=0,jmax_s=0;
       for(int i=0;i<Nx/scale[n];i+=stride)
       {
          for(int j=0;j<Ny/scale[n];j+=stride)
          {
             int h=0;
             for(int k=0;k<indx_s[n].size();k++)
             {
                if(i+indx_s[n][k]>0 && i+indx_s[n][k]<Nx && j+indy_s[n][k]>0 && j+indy_s[n][k]<Ny)
                {
                   if(abs(w_s[n][k]-s_p[i+indx_s[n][k]][j+indy_s[n][k]])<sens*err[(int)w[n][k]])//25)
                      h+=1;
                   else
                      h+=-1;
                }
             }
             if(h>hmax_s) {hmax_s=h;imax_s=i;jmax_s=j;}
          }
       }
       if(hmax_s>sens1*w_s[n].size()) 
       {
       int hmax=0,imax=0,jmax=0;
       for(int i=imax_s*scale[n]-scale[n]/2;i<imax_s*scale[n]+scale[n]/2;i+=stride)
       {
          for(int j=jmax_s*scale[n]-scale[n]/2;j<jmax_s*scale[n]+scale[n]/2;j+=stride)
          {
             int h=0;
             for(int k=0;k<indx[n].size();k++)
             {
                if(i+indx[n][k]>0 && i+indx[n][k]<Nx && j+indy[n][k]>0 && j+indy[n][k]<Ny)
                {
                   if(abs(w[n][k]-spin[i+indx[n][k]][j+indy[n][k]])<sens*err[(int)w[n][k]])//25)
                      h+=1;
                   else
                      h+=-1;
                }
             }
             if(h>hmax) {hmax=h;imax=i;jmax=j;}
          }
       }
       if(hmax>sens1*w[n].size()) 
       {
          for(int k=0;k<indx[n].size();k++)
          {
             if(imax+indx[n][k]>0 && imax+indx[n][k]<Nx && jmax+indy[n][k]>0 && jmax+indy[n][k]<Ny)
             {
                //out[i+indx[nmax][k]][j+indy[nmax][k]]=spin[i+indx[nmax][k]][j+indy[nmax][k]];
                if(abs(w[n][k]-spin[imax+indx[n][k]][jmax+indy[n][k]])<sens*err[(int)w[n][k]])
                   out[imax+indx[n][k]][jmax+indy[n][k]]=255;
             }
          }
       }
       }    
    }
}
