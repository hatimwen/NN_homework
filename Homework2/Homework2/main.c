/*
Multi-Layer Feed forward Neural Networks
the layer number: L
input number: N_0
output number:N_L
unit number:N_1~N_L-1
*/
#include <stdio.h>
#include "NN_BP.h"


float normal_rand();

int main(){
	// The number of the input layer's nude is set to 'M_input'.
	int M_input = X_LENGTH;
	// The number of each layer's nude is set to 'M'.
	int M = 2;	
	// The number of layer is set to 'L'.
	int L = L_MAX -1;
	int N[L_MAX], *p_N;
	p_N = N;
	for (int i = 0; i < L_MAX; i++){
		N[i] = M;
	}
	N[L] = 1;
	double YY[U_MAX][U_MAX] = { 0 };
	double Y_out[Y_LENGTH] = { 0 };
	double *p_X_data = NULL;
	double *p_D_label = NULL;
	double W_weight[L_MAX][U_MAX][U_MAX];
	double loss = 0;
	char filename[14];
	/* ===================X_data and D_label Samples======= */
	double XX_data[K_MAX][X_LENGTH + 1] = {{1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
	double DD_label[K_MAX][Y_LENGTH] = {{1}, {0}, {0}, {1}};
    /* ====================================================*/



	// One of X_data(e.g. XX_data[cat]) is taken as Test_data every single big loop(totally 4 big loops), 
	// where the rest data is fed as Train_data
	for(int cat = 0; cat < K_MAX; cat++){
		// Firstly, initialize the wights:W
		sprintf(filename, "./output_%d.txt", cat);
		FILE *p = fopen(filename, "w");
		for(int l = 1;l<=L;l++){
			YY[l][0] = 1;
			for(int j =1;j <= N[l];j++){
				for(int i = 0; i <= N[l-1]; i++){
					W_weight[l][j][j] = normal_rand();
					// W_weight[l][j][j] = 0;
				}
			}
		}
		loss = 0;
		// Then, BP starts.
		int kk = 0, k_print_loss = 0;
		double * p_X_test = NULL;
		double *p_D_test = NULL;
		for(int k = 0; k < ITER_MAX; k++){
			k_print_loss = k%K_TRAIN;
			// Load X_data and D_label
			kk = ((k%K_TRAIN) + 1)%K_MAX;
			p_X_data = XX_data[kk];
			p_D_label = DD_label[kk];

			YY[0][0] = 1;
			YY[0][1] = p_X_data[1];
			YY[0][2] = p_X_data[2];
			if(PRINT_FLAG){
				printf("\niter %d", k);
			}
			loss += BP_Learning_Algorithm(Learning_Rate, N, L, W_weight, p_X_data, YY, p_D_label);
			if(k_print_loss == (K_TRAIN -1)){
				fprintf(p, "%lf\n",loss);
				if(PRINT_FLAG){
					printf(" Loss = %lf", loss);
				}
				if(loss < THRESHOLD){
					if(PRINT_FLAG){
						printf("\n******************************Convergence !*************************\n");
					}
					break;
				}
				loss = 0;
			}
		}

		fclose(p);
		// for (int i = 0; i < N[L]; i++){
		// 	printf("***************Y_OUT*************\n");
		// 	printf("N[i] = %d,i =  %d\n", N[i], i);
		// 	printf("%lf\n", Y_out[i]);
		// }

		printf("\n############################ NO.%d Train ##########################\n", cat);
		printf("##########################THE FINAL WEIGHT:######################\n");
		for(int l = 1;l<=L;l++){
			for(int j =1;j <= N[l];j++){
				for(int i = 0; i <= N[l-1]; i++){
					printf("W%d%d_%d = %lf\n", j, i, l, W_weight[l][j][j]);
				}
			}
		}
		// TO DO: Get the result 
		p_X_test = XX_data[cat];
		p_D_test = DD_label[cat];
		for(int l = 1;l<=L;l++){YY[l][0] = 1;}
		YY[0][0] = 1;
		YY[0][1] = p_X_test[1];
		YY[0][2] = p_X_test[2];
		feed_forward_network(N, L, p_X_test, W_weight, YY);
		if(PRINT_FLAG_TEST){
			printf("########################## NO.%d TEST ######################\n", cat);
			printf(" X: %.1f, %.1f,", YY[0][1], YY[0][2]);
			printf(" est = %lf, gt = %.1f\n", YY[L][1], p_D_test[0]);
			printf("#################################################################\n");
		}
		printf("#################################################################\n");
	}
	system("pause");
	return 0;
}

float normal_rand()
{
    static double U, V;
    static int phase = 0;
    double z;
    U = rand() / (RAND_MAX + 1.0);
    V = rand() / (RAND_MAX + 1.0);
    if(phase == 0)
    {
         z = sqrt(-2.0 * log(U))* sin(2.0 * PI * V);
    }
    else
    {
         z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
    }

    phase = 1 - phase;
    return z;
}