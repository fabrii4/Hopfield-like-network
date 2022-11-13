#include <vector>

#ifndef NETWORKLIB_H
#define NETWORKLIB_H

class neuron 
{
  public:
    int L,x,y,s;
    float t; //theta
    neuron (); //initialize empty object
    neuron (int a, int b, int c, int d, float e); //initialize valued object
};

class synapse
{
  public:
    int L1,L2,x1,y1,x2,y2,s;
    float w;
    char* T;
    synapse (); //initialize empty object
    synapse (int a, int b, int c, int d, int e, int f, int g, float h, char* i); //initialize valued object
};

class layer //set of neurons
{
  public:
    int L;
    std::vector<neuron> v; //vector of neurons
    std::vector<std::vector<neuron> > vo; //vector of neurons ordered by (i,j)
    layer (); //initialize empty object
    layer (int a); //initialize valued object
    void add(neuron& n);
    void remove(neuron& n, int k);
    void init(int Lt, int Nx, int Ny);
};

class pattern //set of synapses connecting one layer to one neuron
{
  public:
    int L1,L2,x,y;
    std::vector<synapse> v; //vector of synapses
    std::vector<std::vector<synapse> > vo; //vector of synapses ordered by (i1,j1)
    pattern (); //initialize empty object
    pattern (int a, int b, int c, int d); //initialize valued object
    void add(synapse& w);
    void remove(synapse& w, int k);
    void init(int Lt1,int Lt2, int i2, int j2, int Nx, int Ny);
};

class bundle //set of patterns connecting two layers
{
  public:
    int L1,L2;
    std::vector<pattern> v; //vector of patterns
    std::vector<std::vector<pattern> > vo; //vector of patterns ordered by (i2,j2)
    bundle (); //initialize empty object
    bundle (int a, int b); //initialize valued object
    void add(pattern& p);
    void remove(pattern& p, int k);
    void add_s(synapse& w);
    void remove_s(synapse& w);
    void order();
//    void init(int Lt1, int Lt2, int Nt);
};

class network //set of layers and bundles
{
  public:
    std::vector<layer> v; //vector of layers
    std::vector<layer> vo; //vector of layers ordered by increasing L
    std::vector<bundle> vb; //vector of bundles
    std::vector<std::vector<bundle> > vob; //vector of bundles ordered by increasing L
    void add_n(neuron& n);
    void remove_n(neuron& n);
    void add_s(synapse& w);
    void remove_s(synapse& w);
    void add_l(layer& l);
    void remove_l(layer& l, int k);
    void add_p(pattern& p);
    void remove_p(pattern& p);
    void add_b(bundle& b);
    void remove_b(bundle& b, int k);
//    void init(int N);
};

#endif
