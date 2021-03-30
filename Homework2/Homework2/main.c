/*
Multi-Layer Feed forward Neural Networks
the layer number: L
input number: N_0
output number:N_L
unit number:N_1~N_L-1
*/
#include "matrix.h"
#include <stdio.h>
#include "NN_BP.h"

#define X_LENGTH 10
#define Y_LENGTH 10
#define Learning_Rate 0.1

int main(){
	// The number of the input layer's nude is set to 'M_input'.
	int M_input = X_LENGTH;
	// The number of each layer's nude is set to 'M'.
	int M = 10;	
	// The number of layer is set to 'L'.
	int L = 2;
	int N[L_MAX], *p_N;
	p_N = N;
	for (int i = 0; i < L_MAX; i++){
		N[i] = M;
	}
	double YY[U_MAX][U_MAX] = { 0 };
	double Y_out[Y_LENGTH] = { 0 }, *p_Y_out;
	p_Y_out = Y_out;
	double X_data[X_LENGTH] = { 0 }, *p_X_data;
	p_X_data = X_data;
	double D_label[Y_LENGTH] = { 0 };
	/* ================TO DO: X_data = dataloader();======= */

	for (int i = 0; i < M_input; i++){
		printf("%lf\n",X_data[i]);
	}

	// Firstly, initialize the wights:W
	double W_weight[L_MAX][U_MAX][U_MAX];
	BP_Learning_Algorithm(Learning_Rate, N, L, W_weight, X_data, YY, D_label);

	for (int i = 0; i < N[L]; i++){
		printf("***************Y_OUT*************\n");
		printf("N[i] = %d,i =  %d\n", N[i], i);
		printf("%lf\n", Y_out[i]);
	}

	system("pause");
	return 0;
}