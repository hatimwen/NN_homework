/*
Multi-Layer Feed forward Neural Networks
the layer number: L
input number: N_0
output number:N_L
unit number:N_1~N_L-1
*/
#include "NN_BP.h"


float normal_rand();

int main(){
	// The number of layer is set to 'L'.
	int L = L_MAX -1;
	int N[L_MAX], *p_N;
	p_N = N;
	N[0] = X_LENGTH;
	// Hidden layer:
	for (int i = 1; i < L; i++){
		N[i] = U_MAX - 1;
	}
	N[L] = Y_LENGTH;
	double YY[U_MAX][U_MAX] = { 0 };
	double Y_out[Y_LENGTH] = { 0 };
	double *p_X_data = NULL;
	double *p_D_label = NULL;
	double W_weight[L_MAX][U_MAX][U_MAX];
	double loss = 0;
	char filename[19];
	/* ===================X_data and D_label Samples======= */
	double XX_data[K_MAX][X_LENGTH + 1];// = {{1, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}};
	double DD_label[K_MAX][Y_LENGTH];// = {{1}, {0}, {0}, {1}};
	FILE *p_data;
	p_data = fopen("./Parity_Check_data_7.txt", "r");
	if(p_data==NULL)    //打开文件失败
		{  
			printf("File cannot open! ");
		}
	char strline[X_LENGTH + 2];
    char*p = NULL;
	int i = 0;
	while(!feof(p_data)){
		p = fgets(strline, X_LENGTH + 3, p_data);
        if(p == NULL){fclose(p_data);}
		else{
			XX_data[i][0] = 1;
			DD_label[i][0] = strline[X_LENGTH] - '0';
			for(int j =0; j < X_LENGTH; j++){
				XX_data[i][j+1] = strline[j] - '0';
				// printf("XX_data[%d][%d] = %f\n",i, j+1, XX_data[i][j+1]);
			}
			// printf("DD_label[%d][0] = %f\n",i, DD_label[i][0]);
			i++;
		}

	}
	fclose(p_data);
    /* ====================================================*/



	// One of X_data(e.g. XX_data[cat]) is taken as Test_data every single big loop(totally 4 big loops), 
	// where the rest data is fed as Train_data
	// for(int cat = K_MAX -1; cat < K_MAX; cat++){
		// Firstly, initialize the wights:W
		sprintf(filename, "output_7_train.txt");
		FILE *pf = fopen(filename, "w");
		for(int l = 1;l<=L;l++){
			YY[l][0] = 1;
			for(int j =1;j <= N[l];j++){
				for(int i = 0; i <= N[l-1]; i++){
					srand(i);
					W_weight[l][j][i] = normal_rand();
					// W_weight[l][j][i] = 0;
				}
			}
		}
		// loss = 0;
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
			for(int d = 1; d<X_LENGTH+1;d++){
				YY[0][d] = p_X_data[d];
			}
			// YY[0][1] = p_X_data[1];
			// YY[0][2] = p_X_data[2];
			if(PRINT_FLAG){
				printf("\niter %d", k);
			}
			loss += BP_Learning_Algorithm(Learning_Rate, N, L, W_weight, p_X_data, YY, p_D_label);
			if(k_print_loss == (K_TRAIN -1)){
				fprintf(pf, "%lf\n",loss);// / (double)(K_TRAIN));
				if(PRINT_FLAG){
					printf(" Loss = %lf\n", loss);// / (double)(K_TRAIN));
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

		fclose(pf);
		// for (int i = 0; i < N[L]; i++){
		// 	printf("***************Y_OUT*************\n");
		// 	printf("N[i] = %d,i =  %d\n", N[i], i);
		// 	printf("%lf\n", Y_out[i]);
		// }

		// printf("\n############################ NO.%d Train ##########################\n", cat);
		printf("\n############################### Train ##########################\n");
		printf("##########################THE FINAL WEIGHT:######################\n");
		for(int l = 1;l<=L;l++){
			for(int j =1;j <= N[l];j++){
				for(int i = 0; i <= N[l-1]; i++){
					printf("W%d%d_%d = %lf\n", j, i, l, W_weight[l][j][i]);
				}
			}
		}
		// Compute the success rate of testing:
		double loss_test = 0;
		double num_success = 0;
		double success_rate_test = 0;
		for(int cat = K_TRAIN; cat < K_MAX ; cat ++){
			p_X_test = XX_data[cat];
			p_D_test = DD_label[cat];
			for(int l = 1;l<=L;l++){YY[l][0] = 1;}
			YY[0][0] = 1;
			for(int d = 1; d<X_LENGTH+1;d++){
				YY[0][d] = p_X_test[d];
			}
			// YY[0][1] = p_X_test[1];
			// YY[0][2] = p_X_test[2];
			feed_forward_network(N, L, p_X_test, W_weight, YY);
			loss_test += 0.5 * pow((YY[L][1] - p_D_test[0]), 2);
			// printf("(int)YY[L][1] = %d, int)p_D_test[0] = %d\n", (int)(YY[L][1]+0.5), (int)p_D_test[0]);
			// ps. C语言的取整是向下。。
			if((int)(YY[L][1]+0.5) == (int)p_D_test[0]){
				num_success += 1;
			}
			if(PRINT_FLAG_TEST){
				printf("########################## NO.%d TEST ######################\n", cat);
				// printf(" X: %.1f, %.1f,", YY[0][1], YY[0][2]);
				printf("X: ");
				for(int d = 1;d<X_LENGTH+1; d++){
					printf("%.1f ", YY[0][d]);
				}
				printf(" est = %lf, gt = %.1f\n", YY[L][1], p_D_test[0]);
				printf("#################################################################\n");
			}
		}
		loss_test = loss_test / (K_MAX - K_TRAIN);
		printf("#################################################################\n");
		printf("LOSS_TEST = %lf\n", loss_test);
		success_rate_test = num_success / (K_MAX - K_TRAIN);
		printf("success_rate_test = %lf\n", success_rate_test);
		printf("#################################################################\n");


		// Compute the success rate of training:
		double loss_train = 0, success_rate_train = 0;
		for(int cat = 0; cat < K_TRAIN ; cat ++){
			p_X_test = XX_data[cat];
			p_D_test = DD_label[cat];
			for(int l = 1;l<=L;l++){YY[l][0] = 1;}
			YY[0][0] = 1;
			for(int d = 1; d<X_LENGTH+1;d++){
				YY[0][d] = p_X_test[d];
			}
			// YY[0][1] = p_X_test[1];
			// YY[0][2] = p_X_test[2];
			feed_forward_network(N, L, p_X_test, W_weight, YY);
			loss_train += 0.5 * pow((YY[L][1] - p_D_test[0]), 2);
			if((int)(YY[L][1]+0.5) == (int)p_D_test[0]){
				success_rate_train += 1;
			}
		}
		printf("#################################################################\n");
		loss_train = loss_train / K_TRAIN;
		printf("LOSS_TRAIN = %lf\n", loss_train);
		
		success_rate_train = success_rate_train / K_TRAIN;
		printf("success_rate_train = %lf\n", success_rate_train);
		printf("#################################################################\n");
	// }
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