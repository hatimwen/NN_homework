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
	double YY[U_MAX][U_MAX] = { 0 }; //, **p_YY;
	// p_YY = YY;
	double Y_out[Y_LENGTH] = { 0 }; //, *p_Y_out;
	// p_Y_out = Y_out;
	double X_data[X_LENGTH] = { 0 }, *p_X_data;
	p_X_data = X_data;
	double D_label[Y_LENGTH] = { 0 }, *p_D_label;
	p_D_label = D_label;
	/* ================TO DO: X_data = dataloader();======= */
	double XX_data[4][3] = {{1, 1,0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
	double DD_label[4][1] = {{1}, {0}, {0}, {1}};

    /* ====================================================*/
	for (int i = 0; i < M_input; i++){
		printf("%lf\n",X_data[i]);
	}

	// Firstly, initialize the wights:W
	double W_weight[L_MAX][U_MAX][U_MAX];
	double loss = 0;
    FILE *p = fopen("./output.txt", "w");
	for(int l = 1;l<=L;l++){
		YY[l][0] = 1;
		for(int j =1;j <= N[l];j++){
			for(int i = 0; i <= N[l-1]; i++){
				W_weight[l][j][j] = normal_rand();
			}
		}
	}
	for(int k = 0; k < ITER_MAX; k++){
		int kk = k%4;
		p_X_data = XX_data[kk];
		p_D_label = DD_label[kk];
		double *p_DD;
		p_DD = DD_label;

		YY[0][0] = 1;
		YY[0][1] = p_X_data[1];
		YY[0][2] = p_X_data[2];
		printf("\niter %d", k);
		loss += BP_Learning_Algorithm(Learning_Rate, N, L, W_weight, p_X_data, YY, p_D_label);
		if(kk == 3){
			fprintf(p, "%d %lf\n", k/4, loss);
			printf(" Loss = %lf", loss);
			if(loss < THRESHOLD){
				printf("\n******************************Convergence !*************************\n");
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

	printf("\n##########################THE FINAL WEIGHT:######################\n");
	for(int l = 1;l<=L;l++){
		for(int j =1;j <= N[l];j++){
			for(int i = 0; i <= N[l-1]; i++){
				printf("W%d%d_%d = %lf\n", j, i, l, W_weight[l][j][j]);
			}
		}
	}

	printf("#################################################################\n");
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