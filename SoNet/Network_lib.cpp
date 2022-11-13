#include <vector>
#include <iostream>
#include "Network_lib.hpp"


using namespace std;

//////////////////////////////////////////////
//neuron functions
//////////////////////////////////////////////
neuron::neuron () {} //initialize empty object
neuron::neuron (int a, int b, int c, int d, float e) //initialize valued object
{
   L=a; x=b; y=c; s=d; t=e;
}


//////////////////////////////////////////////
//synapse functions
//////////////////////////////////////////////
synapse::synapse () {} //initialize empty object
synapse::synapse (int a, int b, int c, int d, int e, int f, int g, float h, char* i) //initialize valued object
{
   L1=a; L2=b; x1=c; y1=d; x2=e; y2=f; s=g; w=h; T=i;
}


//////////////////////////////////////////////
//layer functions
//////////////////////////////////////////////
layer::layer () {} //initialize empty object
layer::layer (int a) //initialize valued object
{
   L=a;
}

//add neuron to layer
void layer::add(neuron& n)
{
   if((v.size()>0 && L==n.L)||v.size()==0)
   {
      L=n.L;
      //insert neuron in v
      v.push_back(n);
      //insert neuron in ordered vector vo
      if(vo.size()<=n.x)
      {
         if(vo.size()>0 && vo[0].size()<=n.y)
         {
            for(int i=0;i<vo.size();i++)
            {
               vo[i].resize(n.y+1);
            }
         }
         vector<neuron> nvj;
         nvj.resize(n.y+1);
         vo.resize(n.x+1,nvj);
      }
      else if(vo[0].size()<=n.y)
      {
         for(int i=0;i<vo.size();i++)
         {
            vo[i].resize(n.y+1);
         }
      }
      vo[n.x][n.y]=n;
   }
}

//remove neuron from layer
void layer::remove(neuron& n, int k=-1)
{
   if(k==-1)
   {
      for(int i=0;i<v.size();i++)
      {
         if(v[i].x==n.x && v[i].y==n.y && v[i].L==n.L)
         {
            v.erase(v.begin() + i); 
            neuron nt;
            vo[n.x][n.y]=nt;
            break;
         }
      }
   }
   else
   {
      neuron nt;
      vo[v[k].x][v[k].y]=nt;
      v.erase(v.begin() + k);
   }
}

//initialize layer
void layer::init(int Lt, int Nx, int Ny)
{
   v.clear(); vo.clear();
   L=Lt;
   for(int i=0;i<Nx;i++)
   {
      vector<neuron> vj;
      for(int j=0;j<Ny;j++)
      {
         neuron n (Lt,i,j,0, 0);
         v.push_back(n);
         vj.push_back(n);
      }
      vo.push_back(vj);
   }
}


//////////////////////////////////////////////
//pattern functions
//////////////////////////////////////////////
pattern::pattern () {} //initialize empty object
pattern::pattern (int a, int b, int c, int d) //initialize valued object
{
   L1=a; L2=b; x=c; y=d;
}

//add synapse to pattern
void pattern::add(synapse& w)
{
   if((w.x2==x && w.y2==y && L1==w.L1 && L2==w.L2) || v.size()<1)
   {
      L1=w.L1; L2=w.L2; x=w.x2; y=w.y2;
      //insert synapse in v
      v.push_back(w);
      //insert synapse in ordered vector vo
      if(vo.size()<=w.x1)
      {
         if(vo.size()>0 && vo[0].size()<=w.y1)
         {
            for(int i=0;i<vo.size();i++)
            {
               vo[i].resize(w.y1+1);
            }
         }
         vector<synapse> wvj;
         wvj.resize(w.y1+1);
         vo.resize(w.x1+1,wvj);
      }
      else if(vo[0].size()<=w.y1)
      {
         for(int i=0;i<vo.size();i++)
         {
            vo[i].resize(w.y1+1);
         }
      }
      vo[w.x1][w.y1]=w;
   }
}

//remove synapse from pattern
void pattern::remove(synapse& w, int k=-1)
{
   if(k==-1)
   {
      for(int i=0;i<v.size();i++)
      {
         if(v[i].x1==w.x1 && v[i].y1==w.y1 && v[i].x2==w.x2 && v[i].y2==w.y2 && v[i].L1==w.L1 && v[i].L2==w.L2)
         {
            v.erase(v.begin() + i); 
            synapse wt;
            vo[w.x1][w.y1]=wt;
            break;
         }
      }
   }
   else
   {
      synapse wt;
      vo[v[k].x1][v[k].y1]=wt;
      v.erase(v.begin() + k);
   }
}

//initialize pattern
void pattern::init(int Lt1,int Lt2, int i2, int j2, int Nx, int Ny)
{
   v.clear(); vo.clear();
   L1=Lt1; L2=Lt2; x=i2; y=j2;
   for(int i=0;i<Nx;i++)
   {
      vector<synapse> vj;
      for(int j=0;j<Ny;j++)
      {
         synapse wt (L1,L2,i,j,x,y,0,0,(char*)"T");
         v.push_back(wt);
         vj.push_back(wt);
      }
      vo.push_back(vj);
   }
}


//////////////////////////////////////////////
//bundle functions
//////////////////////////////////////////////
bundle::bundle () {} //initialize empty object
bundle::bundle (int a, int b) //initialize valued object
{
   L1=a; L2=b;
}

//add pattern to bundle
void bundle::add(pattern& p)
{
   if((v.size()>0 && L1==p.L1 && L2==p.L2) || v.size()<1)
   {
      L1=p.L1; L2=p.L2;
      //insert pattern in v
      v.push_back(p);
      //insert pattern in ordered vector vo
      if(vo.size()<=p.x)
      {
         if(vo.size()>0 && vo[0].size()<=p.y)
         {
            for(int i=0;i<vo.size();i++)
            {
               vo[i].resize(p.y+1);
            }
         }
         vector<pattern> pvj;
         pvj.resize(p.y+1);
         vo.resize(p.x+1,pvj);
      }
      else if(vo[0].size()<=p.y)
      {
         for(int i=0;i<vo.size();i++)
         {
            vo[i].resize(p.y+1);
         }
      }
      vo[p.x][p.y]=p;
   }
}

//remove pattern from bundle
void bundle::remove(pattern& p, int k=-1)
{
   if(k==-1)
   {
      for(int i=0;i<v.size();i++)
      {
         if(v[i].x==p.x && v[i].y==p.y && v[i].L1==p.L1 && v[i].L2==p.L2)
         {
            v.erase(v.begin() + i); 
            pattern pt;
            vo[p.x][p.y]=pt;
            break;
         }
      }
   }
   else
   {
      pattern pt;
      vo[v[k].x][v[k].y]=pt;
      v.erase(v.begin() + k);
   }
}

//add synapse to bundle
void bundle::add_s(synapse& w)
{
   if(L1==w.L1 && L2==w.L2)
   {
      int f=0;
      for(int i=0;i<v.size();i++)
      {
         if(v[i].x==w.x2 && v[i].y==w.y2)
         {
            f=1;
            v[i].add(w);
            vo[w.x2][w.y2].add(w);
            break;
         }
      }
      if(f==0)
      {
         pattern p (w.L1, w.L2, w.x2, w.y2);
         p.add(w);
         add(p);
      }
   }
}

//remove synapse from bundle
void bundle::remove_s(synapse& w)
{
   if(L1==w.L1 && L2==w.L2)
   {
      for(int i=0;i<v.size();i++)
      {
         if(v[i].x==w.x2 && v[i].y==w.y2)
         {
            v[i].remove(w);
            vo[w.x2][w.y2].remove(w);
            if(v[i].v.size()==0)
               remove(v[i]);
            break;
         }
      }
   }
}


//////////////////////////////////////////////
//network functions
//////////////////////////////////////////////
//add neuron to network
void network::add_n(neuron& n)
{
   int f=0;
   for(int i=0;i<v.size();i++)
   {
      if(v[i].L==n.L)
      {
         f=1;
         v[i].add(n);
         vo[n.L].add(n);
         break;
      }
   }
   if(f==0)
   {
      layer lt (n.L);
      lt.add(n);
      v.push_back(lt);
      if(vo.size()<=n.L)
      {
         vo.resize(n.L+1,lt);
      }
      vo[n.L]=lt;
   }
}

//remove neuron from network
void network::remove_n(neuron& n)
{
   for(int i=0;i<v.size();i++)
   {
      if(v[i].L==n.L)
      {
         v[i].remove(n);
         vo[n.L].remove(n);
         if(v[i].v.size()==0)
            remove_l(v[i],-1);
         break;
      }
   }
}

//add synapse to network
void network::add_s(synapse& w)
{
   int f=0;
   for(int i=0;i<vb.size();i++)
   {
      if(vb[i].L1==w.L1 && vb[i].L2==w.L2)
      {
         f=1;
         vb[i].add_s(w);
         vob[w.L1][w.L2].add_s(w);
         break;
      }
   }
   if(f==0)
   {
      bundle b (w.L1, w.L2);
      b.add_s(w);
      vb.push_back(b);
      if(vob.size()<=w.L1)
      {
         if(vob.size()>0 && vob[0].size()<=w.L2)
         {
            for(int i=0;i<vob.size();i++)
            {
               vob[i].resize(w.L2+1);
            }
         }
         vector<bundle> bvj;
         bvj.resize(w.L2+1);
         vob.resize(w.L1+1,bvj);
      }
      else if(vob[0].size()<=w.L2)
      {
         for(int i=0;i<vob.size();i++)
         {
            vob[i].resize(w.L2+1);
         }
      }
      vob[w.L1][w.L2]=b;
   }
}

//remove synapse from network
void network::remove_s(synapse& w)
{
   for(int i=0;i<vb.size();i++)
   {
      if(vb[i].L1==w.L1 && vb[i].L2==w.L2)
      {
         vb[i].remove_s(w);
         vob[w.L1][w.L2].remove_s(w);
         if(vb[i].v.size()==0)
            remove_b(vb[i],-1);
         break;
      }
   }
}

//add layer to network
void network::add_l(layer& l)
{
   v.push_back(l);
   //insert layer in ordered vector vo
   if(vo.size()<=l.L)
   {
      vo.resize(l.L+1);
   }
   vo[l.L]=l;
}

//remove layer from network
void network::remove_l(layer& l, int k=-1)
{
   if(k==-1)
   {
      for(int i=0;i<v.size();i++)
      {
         if(v[i].L==l.L)
         {
            v.erase(v.begin() + i); 
            layer lt;
            vo[l.L]=lt;
            break;
         }
      }
   }
   else
   {
      layer lt;
      vo[v[k].L]=lt;
      v.erase(v.begin() + k);
   }
}

//add pattern to network
void network::add_p(pattern& p)
{
   int f=0;
   for(int i=0;i<vb.size();i++)
   {
      if(vb[i].L1==p.L1 && vb[i].L2==p.L2)
      {
         f=1;
         vb[i].add(p);
         vob[p.L1][p.L2].add(p);
         break;
      }
   }
   if(f==0)
   {
      bundle b (p.L1, p.L2);
      b.add(p);
      vb.push_back(b);
      if(vob.size()<=p.L1)
      {
         if(vob.size()>0 && vob[0].size()<=p.L2)
         {
            for(int i=0;i<vob.size();i++)
            {
               vob[i].resize(p.L2+1);
            }
         }
         vector<bundle> bvj;
         bvj.resize(p.L2+1);
         vob.resize(p.L1+1,bvj);
      }
      else if(vob[0].size()<=p.L2)
      {
         for(int i=0;i<vob.size();i++)
         {
            vob[i].resize(p.L2+1);
         }
      }
      vob[p.L1][p.L2]=b;
   }
}

//remove pattern from network
void network::remove_p(pattern& p)
{
   for(int i=0;i<vb.size();i++)
   {
      if(vb[i].L1==p.L1 && vb[i].L2==p.L2)
      {
         vb[i].remove(p);
         vob[p.L1][p.L2].remove(p);
         if(vb[i].v.size()==0)
            remove_b(vb[i],-1);
         break;
      }
    }
}

//add bundle to network
void network::add_b(bundle& b)
{
   vb.push_back(b);
   if(vob.size()<=b.L1)
   {
      if(vob.size()>0 && vob[0].size()<=b.L2)
      {
         for(int i=0;i<vob.size();i++)
         {
            vob[i].resize(b.L2+1);
         }
      }
      vector<bundle> bvj;
      bvj.resize(b.L2+1);
      vob.resize(b.L1+1,bvj);
   }
   else if(vob[0].size()<=b.L2)
   {
      for(int i=0;i<vob.size();i++)
      {
         vob[i].resize(b.L2+1);
      }
   }
   vob[b.L1][b.L2]=b;
}

//remove bundle from network
void network::remove_b(bundle& b, int k=-1)
{
   if(k==-1)
   {
      for(int i=0;i<vb.size();i++)
      {
         if(vb[i].L1==b.L1 && vb[i].L2==b.L2)
         {
            vb.erase(vb.begin() + i); 
            bundle bt;
            vob[b.L1][b.L2]=bt;
            break;
         }
      }
   }
   else
   {
      bundle bt;
      vob[vb[k].L1][vb[k].L2]=bt;
      vb.erase(vb.begin() + k);
   }
}
