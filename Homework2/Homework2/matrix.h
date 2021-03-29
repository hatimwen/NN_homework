#ifndef _MATRIX_H_
#define _MATRIX_H_

void matrix_m(double **a_matrix, const double **b_matrix, const double **c_matrix,
	int krow, int kline, int kmiddle, int ktrl)

	//  a_matrix=b_matrix*c_matrix
	//  krow  :行数
	//  kline :列数
	//  ktrl  : 大于0:两个正数矩阵相乘 不大于0:正数矩阵乘以负数矩阵

{
	int k, k2, k4;
	double stmp;

	for (k = 0; k < krow; k++)
	{
		for (k2 = 0; k2 < kline; k2++)
		{
			stmp = 0.0;
			for (k4 = 0; k4 < kmiddle; k4++)
			{
				stmp += b_matrix[k][k4] * c_matrix[k4][k2];
			}
			a_matrix[k][k2] = stmp;
		}
	}
	if (ktrl <= 0)
	{
		for (k = 0; k < krow; k++)
		{
			for (k2 = 0; k2 < kline; k2++)
			{
				a_matrix[k][k2] = -a_matrix[k][k2];
			}
		}
	}
}


#endif