/*
Multi-Layer Feed forward Neural Networks
the layer number: L
input number: N_0
output number:N_L
unit number:N_1~N_L-1
*/
#include "matrix.h"
#include "NN_BP.h"

#define X_LENGTH 10
#define Y_LENGTH 10

int main(){
	// The number of the input layer's nude is set to 'M_input'.
	int M_input = X_LENGTH;
	// The number of each layer's nude is set to 'M'.
	int M = 10;	
	// The number of layer is set to 'L'.
	int L = 2;
	int N[L_MAX];
	for (int i = 0; i < L_MAX; i++){
		N[i] = M;
	}
	double Y_out[Y_LENGTH] = { 0 };
	double X_data[X_LENGTH] = {0};
	/* ================TO DO: X_data = dataloader();======= */

	for (int i = 0; i < M_input; i++){
		printf("%lf\n",X_data[i]);
	}

	BP_Learning_Algorithm(N, L, X_data, Y_out);

	for (int i = 0; i < N[L]; i++){
		printf("N[i] = %d,i =  %d", N[i], i);
		printf("**********************Y_OUT*************\n");
		printf("%lf\n", Y_out[i]);
	}

	// system("pause");
	return 0;
}