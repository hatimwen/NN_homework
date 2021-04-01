#include "NN_BP.h"

double sigmoid(double s){
	return 1.0 / (1.0 + exp(-s));
}

double feed_forward_unit(int N, double W[], double Y[]){

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
		s += W[i] * Y[i];
	}
	y_new = sigmoid(s);
	return y_new;
}

void feed_forward_layer(int* N, int l, double (*W)[U_MAX], double* Y, double* Y_new){
	// Building each (hidden) layer with units.

	// Input: 
	// int* N: N[] consist of N[i] wherer i ranges from 0 to L
	// int l: the 'l' th layer
	// double** W: wji_l consist of wji_l where i ranges from 0 to N[l-1] -1 and j ranges from 0 to N[l] -1
	// double* Y: Y_l-1(k) consist of yi_l-1(k) where i ranges from 1 to N[l-1]

	// Output:
	// double* Y_new: Y_l(k) consist of yj_l(k) where j ranges from 1 to N[l]
	//double Y_new[L_MAX] = {0};
	// Y_new[0] = 1;
	for (int j = 1; j <= N[l]; j++){
		Y_new[j] = feed_forward_unit(N[l - 1], W[j], Y);
	}
	// return Y_new;
}


void feed_forward_network(int* N, int L, double* X_in, double (*W)[U_MAX][U_MAX], double(*YY)[U_MAX]){
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
	//double Y_mid_out[L_MAX] = { 0 }, Y_mid_in[L_MAX] = { 0 }, Y_0[L_MAX] = { 0 };
	// The first layer:
	feed_forward_layer(N, 1, W[1], X_in, YY[1]);
	// Other layers:
	for (int l = 2; l <= L; l++){
		//memcpy(Y_mid_in, Y_mid_out, L_MAX * sizeof(double));
		//memcpy(Y_mid_out, Y_0, L_MAX * sizeof(double));
		feed_forward_layer(N, l, W[l], YY[l - 1], YY[l]);

	}
	//memcpy(Y_out[L], Y_mid_out, L_MAX * sizeof(double));
	//return Y_out;
}


void feed_back_layer(double lr, int* N, int l, double(*YY)[U_MAX], double * Delta_1, double(*W)[U_MAX][U_MAX]){
	// Needed for layers except the last layer: yj_l, deltaq_l+1, wqj_l+1
	// double* Y: Y_l(k) consist of yj_l(k) where j ranges from 0 to N[l]

	// Output:
	// Delta_0: j ranges from 1 to N[l],  the length of which is the same as that of Nodes.
	double Delta_0[U_MAX] = { 0 };
	double W_delta[U_MAX][U_MAX] = { 0 };
	for (int j = 1; j <= N[l]; j++){
		double sum_q = 0;
		for (int q = 0; q <= N[l + 1]; q++){
			sum_q += Delta_1[q] * W[l + 1][q][j];
		}
		Delta_0[j] = YY[l][j] * (1.0 - YY[l][j]) * sum_q;
		for (int i = 0; i <= N[l - 1]; i++){
			W_delta[j][i] = Delta_0[j] * YY[l - 1][i];
			W[l][j][i] += lr * W_delta[j][i];
		}
	}
	//return Delta_0;

}

double BP_Learning_Algorithm(double lr, int* N, int L, double(*W_weight)[U_MAX][U_MAX], double* X_data, double(*YY)[U_MAX], double* D_label){
	// printf("***********************BP STARTS******************\n");
	// Input:
	// int* N: N[] consist of N[i] wherer i ranges from 0 to L
	// int L: the number of the layers

	// Output:
	//  ---------- THE FINAL OUTPUT OF THIS NETWORK----------
	// double* YY: Y_L(k) consist of yi_L(k) where i ranges from 0 to N[L]

	// Secondly, input X_data and compute the outputs YY
	feed_forward_network(N, L, X_data, W_weight, YY);
	// double loss = fabs(YY[L][1] - D_label[0]);
	double loss = 0.5 * pow((YY[L][1] - D_label[0]), 2);
	if(PRINT_FLAG){
		printf(" X: %.1f, %.1f,", YY[0][1], YY[0][2]);
		printf(" est = %lf, gt = %.1f", YY[L][1], D_label[0]);
	}
	// Thirdly, Compute Back-Propagation-Errors
	// 1) The last layer:
	double Delta_all[L_MAX][U_MAX] = { 0 };
	//double Delta_L[U_MAX] = { 0 };
	double W_delta_L[U_MAX][U_MAX] = { 0 };
	for (int j = 1; j <= N[L]; j++){		// N[L] = 1, output layer
		Delta_all[L][j] = YY[L][j] * (1.0 - YY[L][j]) * (D_label[j-1] - YY[L][j]);
		for (int i = 0; i <= N[L - 1]; i++){
			W_delta_L[j][i] = Delta_all[L][j] * YY[L - 1][i];
			W_weight[L][j][i] += lr * W_delta_L[j][i];
		}
	}
	// 2) The other layers:
	for (int l = L - 1; l >= 1; l--){
		feed_back_layer(lr, N, l, YY, Delta_all[l + 1], W_weight);
	}

	// printf("******************************************************************\n");
	// for(int l = 1;l<=L;l++){
	// 	for(int j =1;j <= N[l];j++){
	// 		for(int i = 1; i <= N[l-1]; i++){
	// 			printf("W%d%d_%d = %lf\n", j, i, l, W_weight[l][j][j]);
	// 		}
	// 	}
	// }
	return loss;
}
