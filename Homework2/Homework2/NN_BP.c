#include "NN_BP.h"


double sigmoid(double s){
	return 1 / (1 + exp(s));
}

double feed_forward_unit(int N, double* w, double* y){

	// It's a neural unit(node) that:
	// wji_l, yi_l-1(k) --> sj_l(k)
	// sj_l(k) --> yj_l(k)
	// ***************Input:***************
	// int N: N[l-1]
	// double* w: wj_l[] consist of wji_l where i ranges from 0 to N[l-1]
	// double* y: y_l-1(k)[] consist of yi_l-1(k) where i ranges from 0 to N[l-1]
	// *********Intermediate variable :*********
	// double s: sj_l(k)
	// ***************Output:***************
	// double y_new: yj_l(k)

	double s = 0, y_new = 0;
	for (int i = 0; i <= N; i++){
		s += w[i] * y[i];
	}
	y_new = sigmoid(s);
	return y_new;
}

void feed_forward_layer(int* N, int l, double** W, double* Y, double* Y_new){
	// Building each (hidden) layer with units.

	// Input: 
	// int* N: N[] consist of N[i] wherer i ranges from 0 to L
	// int l: the 'l' th layer
	// double** W: wji_l consist of wji_l where i ranges from 0 to N[l-1] and j ranges from 0 to N[l]
	// double* Y: Y_l-1(k) consist of yi_l-1(k) where i ranges from 0 to N[l-1]

	// Output:
	// double* Y_new: Y_l(k) consist of yi_l(k) where i ranges from 0 to N[l]
	//double Y_new[L_MAX] = {0};
	for (int j = 0; j <= N[l]; j++){
		Y_new[j] = feed_forward_unit(N[l - 1], W[j], Y);
	}
	// return Y_new;
}


void feed_forward_network(int* N, int L, double* X_in, double*** W, double* Y_out){
	// This is a whole network combing all of the layers.

	// Input:
	// int* N: N[] consist of N[i] wherer i ranges from 0 to L
	// int L: the number of the layers
	// double*** W: the whole WEIGHT matrix of the network, and its form: W[l][j][i]
	// -----------THE FIRST INPUT OF THIS NETWORK------------
	// double* X_in: X(k) consist of xi(k) where i ranges from 0 to N[0]

	// Output:
	//  ---------- THE FINAL OUTPUT OF THIS NETWORK----------
	// double* Y_out: Y_L(k) consist of yi_L(k) where i ranges from 0 to N[L]

	// Y_mid_out: the intermediate matrix
	double Y_mid_out[L_MAX] = { 0 }, Y_mid_in[L_MAX] = { 0 }, Y_0[L_MAX] = { 0 };
	// The first layer:
	feed_forward_layer(N, 1, W[0], X_in, Y_mid_out);
	// Other layers:
	for (int l = 2; l <= L; l++){
		memcpy(Y_mid_in, Y_mid_out, L_MAX * sizeof(double));
		memcpy(Y_mid_out, Y_0, L_MAX * sizeof(double));
		feed_forward_layer(N, l, W[l], Y_mid_in, Y_mid_out);

	}
	memcpy(Y_out, Y_mid_out, L_MAX * sizeof(double));
	//return Y_out;
}


void BP_Learning_Algorithm(int* N, int L, double* X_data, double* Y_out){

	// Input:
	// int* N: N[] consist of N[i] wherer i ranges from 0 to L
	// int L: the number of the layers

	// Output:
	//  ---------- THE FINAL OUTPUT OF THIS NETWORK----------
	// double* Y_out: Y_L(k) consist of yi_L(k) where i ranges from 0 to N[L]

	// Firstly, initialize the wights:W
	double W_wight[L_MAX][U_MAX][U_MAX];

	// Secondly, input X_data and compute the outputs Y_out
	feed_forward_network(N, L, X_data, W_wight, Y_out);
}
