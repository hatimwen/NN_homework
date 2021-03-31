#ifndef _NN_BP_H_
#define _NN_BP_H_


#include <math.h> 
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.141592654
#define X_LENGTH 2
#define Y_LENGTH 2
#define Learning_Rate 1
#define ITER_MAX 1000000
#define THRESHOLD 0.00001
// the max number of layer, and N[l] remains the number of the nodes of the 'l' th layer !!!! L_MAX = L +1
#define L_MAX 3 

// the max number of neural unit(node) of each layer, which decides the range of i and j !!! U_MAX = U +1
#define U_MAX 3
double sigmoid(double s);
double feed_forward_unit(int N, double w[], double y[]);
void feed_forward_layer(int* N, int l, double (*W)[U_MAX], double* Y, double* Y_new);
void feed_forward_network(int* N, int L, double* X_in, double (*W)[U_MAX][U_MAX], double(*YY)[U_MAX]);

void feed_back_layer(double lr, int* N, int l, double(*YY)[U_MAX], double * Delta_1, double(*W)[U_MAX][U_MAX]);

double BP_Learning_Algorithm(double lr, int* N, int L, double(*W_weight)[U_MAX][U_MAX], double* X_data, double(*YY)[U_MAX], double* D_label);

#endif