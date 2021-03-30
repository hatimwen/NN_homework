#ifndef _NN_BP_H_
#define _NN_BP_H_


#include <math.h> 
#include <string.h>

// the max number of layer, and N[l] remains the number of the nodes of the 'l' th layer
#define L_MAX 100 

// the max number of neural unit(node) of each layer, which decides the range of i and j
#define U_MAX 100
double sigmoid(double s);
double feed_forward_unit(int N, double w[], double y[]);
void feed_forward_layer(int* N, int l, double(*W)[U_MAX], double* Y, double* Y_new);
void feed_forward_network(int* N, int L, double* X_in, double (*W)[U_MAX][U_MAX], double** YY);

void feed_back_layer(double lr, int* N, int l, double ** YY, double * Delta_1, double(*W)[U_MAX][U_MAX]);

void BP_Learning_Algorithm(double lr, int* N, int L, double(*W_weight)[U_MAX][U_MAX], double* X_data, double** YY, double* D_label);

#endif