#include "Gravity_GPU.cuh"
#include <device_launch_parameters.h>//threadIdx.x
#include <cstdio>
#include <iostream>
#include<thrust/reduce.h>
#include <thrust/transform_reduce.h>   
//ceshi 
# include <thrust/copy.h>
#include <thrust/host_vector.h>
extern __constant__  double pi_gpu = 3.1415926; //圆周率
__device__ int  g_size_gpu;
__device__ int s_size_gpu;
__device__ double p1_gpu;
__constant__  double p2_gpu = 0;//水下
__device__ int n_gpu;
__device__ int g_row_gpu;
__device__ int g_col_gpu;
__device__ double Dif_Lat_gpu;
__device__ double Dif_Lon_gpu;
__device__ int  skip_gpu;
__device__ double  x_gpu[34];
__device__ double  A_gpu[34];

//重载double原子函数
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif



void Gravity::SetGroundPoints(std::vector<PointXYZ> GroundPoints)
{
	g_size = GroundPoints.size();
	GroundPoints_ = (PointXYZ*)malloc(sizeof(PointXYZ) * g_size);
	for (int i = 0; i < GroundPoints.size(); i++)
	{
		GroundPoints_[i] = GroundPoints[i];
	}
}

void Gravity::SetSatellitePoints(std::vector<PointXYZ> SatellitePoints)
{
	s_size = SatellitePoints.size();
	SatellitePoints_ = (PointXYZ*)malloc(sizeof(PointXYZ) * s_size);
	for (int i = 0; i < s_size; i++)
	{
		SatellitePoints_[i] = SatellitePoints[i];
	}
}

//void Gravity::SetRowandCol(const unsigned row__, const unsigned col__)
//{
//	g_row_ = row__;
//	g_col_ = col__;
//}

void Gravity::SetDensity(const double p)
{
	p_1 = p;
}

void Gravity::SetSplit()
{
	//定义一个cuda的设备属性结构体
	cudaDeviceProp prop;
	//获取第1个gpu设备的属性信息
	cudaGetDeviceProperties(&prop, 0);
	//判断数据量是否超过了device所允许的最大线程数
	size_t total = g_size * s_size;//总数据量
	for (split_ = 1; split_ < g_size; split_++)
	{
		if ((prop.maxGridSize[0] > (total / split_)) && is_Integer((double)g_size / split_))
			break;
	}
}

bool Gravity::is_Integer(double input)
{
	return input == (int)input;
}

void Gravity::Setn(const unsigned n)
{
	n_ = n;
	/*节点系数*/
	if (n == 1)
	{
		double zero_points[1] = { 0.0000000000 };
		double W[1] = { 2.0000000000 };
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 2)
	{
		double zero_points[2] = {
		-0.5773502692,
		0.5773502692
		};
		double W[2] = {
		1.0000000000,
		1.0000000000
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 3)
	{
		double zero_points[3] = {
	-0.7745966692,
	0.0000000000,
	0.7745966692
		};
		double W[3] = {
		0.5555555556,
		0.8888888889,
		0.5555555556
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 4)
	{
		double zero_points[4] = {
		-0.8611363116,
		-0.3399810436,
		0.3399810436,
		0.8611363116
		};
		double W[4] = {
		0.3478548451,
		0.6521451549,
		0.6521451549,
		0.3478548451
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 5)
	{
		double zero_points[5] = {
		-0.9061798459,
		-0.5384693101,
		0.0000000000,
		0.5384693101,
		0.9061798459
		};
		double W[5] = {
		0.2369268851,
		0.4786286705,
		0.5688888889,
		0.4786286705,
		0.2369268851
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 6)
	{
		double zero_points[6] = {
		-0.9324695142,
		-0.6612093865,
		-0.2386191861,
		0.2386191861,
		0.6612093865,
		0.9324695142
		};
		double W[6] = {
		0.1713244924,
		0.3607615730,
		0.4679139346,
		0.4679139346,
		0.3607615730,
		0.1713244924
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 7)
	{
		double zero_points[7] = {
		-0.9491079123,
		-0.7415311856,
		-0.4058451514,
		0.0000000000,
		0.4058451514,
		0.7415311856,
		0.9491079123
		};
		double W[7] = {
		0.1294849662,
		0.2797053915,
		0.3818300505,
		0.4179591837,
		0.3818300505,
		0.2797053915,
		0.1294849662
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 8)
	{
		double zero_points[8] = {
		-0.9602898565,
		-0.7966664774,
		-0.5255324099,
		-0.1834346425,
		0.1834346425,
		0.5255324099,
		0.7966664774,
		0.9602898565
		};
		double W[8] = {
		0.1012285363,
		0.2223810345,
		0.3137066459,
		0.3626837834,
		0.3626837834,
		0.3137066459,
		0.2223810345,
		0.1012285363
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 9)
	{
		double zero_points[9] = {
		 -0.9681602395,
		 -0.8360311073,
		 -0.6133714327,
		 -0.3242534234,
		 0.0000000000,
		 0.3242534234,
		 0.6133714327,
		 0.8360311073,
		 0.9681602395
		};
		double W[9] = {
		0.0812743884,
		0.1806481607,
		0.2606106964,
		0.3123470770,
		0.3302393550,
		0.3123470770,
		0.2606106964,
		0.1806481607,
		0.0812743884
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 10)
	{
		double zero_points[10] = {
		-0.9739065285,
		-0.8650633667,
		-0.6794095683,
		-0.4333953941,
		-0.1488743390,
		0.1488743390,
		0.4333953941,
		0.6794095683,
		0.8650633667,
		0.9739065285
		};
		double W[10] = {
		0.0666713443,
		0.1494513492,
		0.2190863625,
		0.2692667193,
		0.2955242247,
		0.2955242247,
		0.2692667193,
		0.2190863625,
		0.1494513492,
		0.0666713443
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 11)
	{
		double zero_points[11] = {
		-0.9782286581,
		-0.8870625998,
		-0.7301520056,
		-0.5190961292,
		-0.2695431560,
		0.0000000000,
		0.2695431560,
		0.5190961292,
		0.7301520056,
		0.8870625998,
		0.9782286581
		};
		double W[11] = {
		0.0556685671,
		0.1255803695,
		0.1862902109,
		0.2331937646,
		0.2628045445,
		0.2729250868,
		0.2628045445,
		0.2331937646,
		0.1862902109,
		0.1255803695,
		0.0556685671
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}
	}
	else if (n == 12)
	{
		double zero_points[12] = {
		-0.9815606342,
		-0.9041172564,
		-0.7699026742,
		-0.5873179543,
		-0.3678314990,
		-0.1252334085,
		0.1252334085,
		0.3678314990,
		0.5873179543,
		0.7699026742,
		0.9041172564,
		0.9815606342
		};
		double W[12] = {
			0.0471753364,
			0.1069393260,
			0.1600783285,
			0.2031674267,
			0.2334925365,
			0.2491470458,
			0.2491470458,
			0.2334925365,
			0.2031674267,
			0.1600783285,
			0.1069393260,
			0.0471753364
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 13)
	{
		double zero_points[13] = {
		-0.9841830547,
		-0.9175983992,
		-0.8015780907,
		-0.6423493394,
		-0.4484927510,
		-0.2304583160,
		0.0000000000,
		0.2304583160,
		0.4484927510,
		0.6423493394,
		0.8015780907,
		0.9175983992,
		0.9841830547
		};
		double W[13] = {
		0.0404840048,
		0.0921214998,
		0.1388735102,
		0.1781459808,
		0.2078160475,
		0.2262831803,
		0.2325515532,
		0.2262831803,
		0.2078160475,
		0.1781459808,
		0.1388735102,
		0.0921214998,
		0.0404840048
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 14)
	{
		double zero_points[14] = {
		-0.9862838087,
		-0.9284348837,
		-0.8272013151,
		-0.6872929048,
		-0.5152486364,
		-0.3191123689,
		-0.1080549487,
		0.1080549487,
		0.3191123689,
		0.5152486364,
		0.6872929048,
		0.8272013151,
		0.9284348837,
		0.9862838087
		};
		double W[14] = {
		0.0351194603,
		0.0801580872,
		0.1215185707,
		0.1572031672,
		0.1855383975,
		0.2051984637,
		0.2152638535,
		0.2152638535,
		0.2051984637,
		0.1855383975,
		0.1572031672,
		0.1215185707,
		0.0801580872,
		0.0351194603
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 15)
	{
		double zero_points[15] = {
		-0.9879925180,
		-0.9372733924,
		-0.8482065834,
		-0.7244177314,
		-0.5709721726,
		-0.3941513471,
		-0.2011940940,
		0.0000000000,
		0.2011940940,
		0.3941513471,
		0.5709721726,
		0.7244177314,
		0.8482065834,
		0.9372733924,
		0.9879925180
		};
		double W[15] = {
		 0.0307532420,
		 0.0703660475,
		 0.1071592205,
		 0.1395706779,
		 0.1662692058,
		 0.1861610000,
		 0.1984314853,
		 0.2025782419,
		 0.1984314853,
		 0.1861610000,
		 0.1662692058,
		 0.1395706779,
		 0.1071592205,
		 0.0703660475,
		 0.0307532420
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 16)
	{
		double zero_points[16] = {
		-0.9894009350,
		-0.9445750231,
		-0.8656312024,
		-0.7554044084,
		-0.6178762444,
		-0.4580167777,
		-0.2816035508,
		-0.0950125098,
		0.0950125098,
		0.2816035508,
		0.4580167777,
		0.6178762444,
		0.7554044084,
		0.8656312024,
		0.9445750231,
		0.9894009350
		};
		double W[16] = {
		0.0271524594,
		0.0622535239,
		0.0951585117,
		0.1246289713,
		0.1495959888,
		0.1691565194,
		0.1826034150,
		0.1894506105,
		0.1894506105,
		0.1826034150,
		0.1691565194,
		0.1495959888,
		0.1246289713,
		0.0951585117,
		0.0622535239,
		0.0271524594
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 17)
	{
		double zero_points[17] = {
		-0.9905754753,
		-0.9506755218,
		-0.8802391537,
		-0.7815140039,
		-0.6576711592,
		-0.5126905371,
		-0.3512317635,
		-0.1784841815,
		0.0000000000,
		0.1784841815,
		0.3512317635,
		0.5126905371,
		0.6576711592,
		0.7815140039,
		0.8802391537,
		0.9506755218,
		0.9905754753
		};
		double W[17] = {
		0.0241483029,
		0.0554595294,
		0.0850361483,
		0.1118838472,
		0.1351363685,
		0.1540457611,
		0.1680041022,
		0.1765627054,
		0.1794464704,
		0.1765627054,
		0.1680041022,
		0.1540457611,
		0.1351363685,
		0.1118838472,
		0.0850361483,
		0.0554595294,
		0.0241483029
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 18)
	{
		double zero_points[18] = {
		-0.9915651684,
		-0.9558239496,
		-0.8926024665,
		-0.8037049590,
		-0.6916870431,
		-0.5597708311,
		-0.4117511615,
		-0.2518862257,
		-0.0847750130,
		0.0847750130,
		0.2518862257,
		0.4117511615,
		0.5597708311,
		0.6916870431,
		0.8037049590,
		0.8926024665,
		0.9558239496,
		0.9915651684
		};
		double W[18] = {
		0.0216160135,
		0.0497145489,
		0.0764257303,
		0.1009420441,
		0.1225552067,
		0.1406429147,
		0.1546846751,
		0.1642764837,
		0.1691423830,
		0.1691423830,
		0.1642764837,
		0.1546846751,
		0.1406429147,
		0.1225552067,
		0.1009420441,
		0.0764257303,
		0.0497145489,
		0.0216160135
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 19)
	{
		double zero_points[19] = {
		-0.9924068438,
		-0.9602081521,
		-0.9031559036,
		-0.8227146565,
		-0.7209661773,
		-0.6005453047,
		-0.4645707414,
		-0.3165641000,
		-0.1603586456,
		0.0000000000,
		0.1603586456,
		0.3165641000,
		0.4645707414,
		0.6005453047,
		0.7209661773,
		0.8227146565,
		0.9031559036,
		0.9602081521,
		0.9924068438
		};
		double W[19] = {
		0.0194617882,
		0.0448142268,
		0.0690445427,
		0.0914900216,
		0.1115666455,
		0.1287539625,
		0.1426067022,
		0.1527660421,
		0.1589688434,
		0.1610544498,
		0.1589688434,
		0.1527660421,
		0.1426067022,
		0.1287539625,
		0.1115666455,
		0.0914900216,
		0.0690445427,
		0.0448142268,
		0.0194617882
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 20)
	{
		double zero_points[20] = {
	-0.9931285992,
	-0.9639719273,
	-0.9122344283,
	-0.8391169718,
	-0.7463319065,
	-0.6360536807,
	-0.5108670020,
	-0.3737060887,
	-0.2277858511,
	-0.0765265211,
	0.0765265211,
	0.2277858511,
	0.3737060887,
	0.5108670020,
	0.6360536807,
	0.7463319065,
	0.8391169718,
	0.9122344283,
	0.9639719273,
	0.9931285992
		};
		double W[20] = {
	0.0176140071,
	0.0406014298,
	0.0626720483,
	0.0832767416,
	0.1019301198,
	0.1181945320,
	0.1316886384,
	0.1420961093,
	0.1491729865,
	0.1527533871,
	0.1527533871,
	0.1491729865,
	0.1420961093,
	0.1316886384,
	0.1181945320,
	0.1019301198,
	0.0832767416,
	0.0626720483,
	0.0406014298,
	0.0176140071
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 21)
	{
		double zero_points[21] = {
		-0.9937521706,
		-0.9672268386,
		-0.9200993342,
		-0.8533633646,
		-0.7684399635,
		-0.6671388042,
		-0.5516188359,
		-0.4243421202,
		-0.2880213168,
		-0.1455618542,
		0.0000000000,
		0.1455618542,
		0.2880213168,
		0.4243421202,
		0.5516188359,
		0.6671388042,
		0.7684399635,
		0.8533633646,
		0.9200993342,
		0.9672268386,
		0.9937521706
		};
		double W[21] = {
		0.0160172283,
		0.0369537898,
		0.0571344254,
		0.0761001136,
		0.0934444235,
		0.1087972992,
		0.1218314161,
		0.1322689386,
		0.1398873948,
		0.1445244040,
		0.1460811336,
		0.1445244040,
		0.1398873948,
		0.1322689386,
		0.1218314161,
		0.1087972992,
		0.0934444235,
		0.0761001136,
		0.0571344254,
		0.0369537898,
		0.0160172283
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 22)
	{
		double zero_points[22] = {
	-0.9942945855,
	-0.9700604978,
	-0.9269567722,
	-0.8658125777,
	-0.7878168060,
	-0.6944872632,
	-0.5876404035,
	-0.4693558380,
	-0.3419358209,
	-0.2078604267,
	-0.0697392733,
	0.0697392733,
	0.2078604267,
	0.3419358209,
	0.4693558380,
	0.5876404035,
	0.6944872632,
	0.7878168060,
	0.8658125777,
	0.9269567722,
	0.9700604978,
	0.9942945855
		};
		double W[22] = {
		0.0146279953,
		0.0337749016,
		0.0522933352,
		0.0697964684,
		0.0859416062,
		0.1004141444,
		0.1129322961,
		0.1232523768,
		0.1311735048,
		0.1365414983,
		0.1392518729,
		0.1392518729,
		0.1365414983,
		0.1311735048,
		0.1232523768,
		0.1129322961,
		0.1004141444,
		0.0859416062,
		0.0697964684,
		0.0522933352,
		0.0337749016,
		0.0146279953
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 23)
	{
		double zero_points[23] = {
		-0.9947693350,
		-0.9725424712,
		-0.9329710868,
		-0.8767523583,
		-0.8048884016,
		-0.7186613631,
		-0.6196098758,
		-0.5095014778,
		-0.3903010380,
		-0.2641356810,
		-0.1332568243,
		0.0000000000,
		0.1332568243,
		0.2641356810,
		0.3903010380,
		0.5095014778,
		0.6196098758,
		0.7186613631,
		0.8048884016,
		0.8767523583,
		0.9329710868,
		0.9725424712,
		0.9947693350
		};
		double W[23] = {
		0.0134118595,
		0.0309880059,
		0.0480376717,
		0.0642324214,
		0.0792814118,
		0.0929157661,
		0.1048920915,
		0.1149966402,
		0.1230490843,
		0.1289057222,
		0.1324620394,
		0.1336545722,
		0.1324620394,
		0.1289057222,
		0.1230490843,
		0.1149966402,
		0.1048920915,
		0.0929157661,
		0.0792814118,
		0.0642324214,
		0.0480376717,
		0.0309880059,
		0.0134118595
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 24)
	{
		double zero_points[24] = {
		-0.9951872200,
		-0.9747285560,
		-0.9382745520,
		-0.8864155270,
		-0.8200019860,
		-0.7401241916,
		-0.6480936519,
		-0.5454214714,
		-0.4337935076,
		-0.3150426797,
		-0.1911188675,
		-0.0640568929,
		0.0640568929,
		0.1911188675,
		0.3150426797,
		0.4337935076,
		0.5454214714,
		0.6480936519,
		0.7401241916,
		0.8200019860,
		0.8864155270,
		0.9382745520,
		0.9747285560,
		0.9951872200
		};
		double W[24] = {
	0.0123412298,
	0.0285313886,
	0.0442774388,
	0.0592985849,
	0.0733464814,
	0.0861901615,
	0.0976186521,
	0.1074442701,
	0.1155056681,
	0.1216704729,
	0.1258374563,
	0.1279381953,
	0.1279381953,
	0.1258374563,
	0.1216704729,
	0.1155056681,
	0.1074442701,
	0.0976186521,
	0.0861901615,
	0.0733464814,
	0.0592985849,
	0.0442774388,
	0.0285313886,
	0.0123412298
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 25)
	{
		double zero_points[25] = {
	-0.9955569698,
	-0.9766639215,
	-0.9429745712,
	-0.8949919979,
	-0.8334426288,
	-0.7592592630,
	-0.6735663685,
	-0.5776629302,
	-0.4730027314,
	-0.3611723058,
	-0.2438668837,
	-0.1228646926,
	0.0000000000,
	0.1228646926,
	0.2438668837,
	0.3611723058,
	0.4730027314,
	0.5776629302,
	0.6735663685,
	0.7592592630,
	0.8334426288,
	0.8949919979,
	0.9429745712,
	0.9766639215,
	0.9955569698
		};
		double W[25] = {
		  0.0113937985,
		  0.0263549866,
		  0.0409391567,
		  0.0549046960,
		  0.0680383338,
		  0.0801407003,
		  0.0910282620,
		  0.1005359491,
		  0.1085196245,
		  0.1148582591,
		  0.1194557635,
		  0.1222424430,
		  0.1231760537,
		  0.1222424430,
		  0.1194557635,
		  0.1148582591,
		  0.1085196245,
		  0.1005359491,
		  0.0910282620,
		  0.0801407003,
		  0.0680383338,
		  0.0549046960,
		  0.0409391567,
		  0.0263549866,
		  0.0113937985
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 26)
	{
		double zero_points[26] = {
			 -0.9958857011,
			 -0.9783854460,
			 -0.9471590667,
			 -0.9026378620,
			 -0.8454459428,
			 -0.7763859488,
			 -0.6964272604,
			 -0.6066922930,
			 -0.5084407148,
			 -0.4030517551,
			 -0.2920048395,
			 -0.1768588204,
			 -0.0592300934,
			 0.0592300934,
			 0.1768588204,
			 0.2920048395,
			 0.4030517551,
			 0.5084407148,
			 0.6066922930,
			 0.6964272604,
			 0.7763859488,
			 0.8454459428,
			 0.9026378620,
			 0.9471590667,
			 0.9783854460,
			 0.9958857011
		};
		//Gauss Weights
		double W[26] = {
		0.0105513726,
		0.0244178511,
		0.0379623833,
		0.0509758253,
		0.0632740463,
		0.0746841498,
		0.0850458943,
		0.0942138004,
		0.1020591611,
		0.1084718405,
		0.1133618165,
		0.1166604435,
		0.1183214153,
		0.1183214153,
		0.1166604435,
		0.1133618165,
		0.1084718405,
		0.1020591611,
		0.0942138004,
		0.0850458943,
		0.0746841498,
		0.0632740463,
		0.0509758253,
		0.0379623833,
		0.0244178511,
		0.0105513726
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 27)
	{
		double zero_points[27] = {
		-0.9961792629,
		-0.9799234760,
		-0.9509005578,
		-0.9094823207,
		-0.8562079080,
		-0.7917716391,
		-0.7170134737,
		-0.6329079719,
		-0.5405515646,
		-0.4411482518,
		-0.3359939036,
		-0.2264593654,
		-0.1139725856,
		0.0000000000,
		0.1139725856,
		0.2264593654,
		0.3359939036,
		0.4411482518,
		0.5405515646,
		0.6329079719,
		0.7170134737,
		0.7917716391,
		0.8562079080,
		0.9094823207,
		0.9509005578,
		0.9799234760,
		0.9961792629
		};
		//Gauss Weights
		double W[27] = {
		0.0097989961,
		0.0226862316,
		0.0352970538,
		0.0474494125,
		0.0589835369,
		0.0697488238,
		0.0796048678,
		0.0884231585,
		0.0960887274,
		0.1025016378,
		0.1075782858,
		0.1112524884,
		0.1134763461,
		0.1142208674,
		0.1134763461,
		0.1112524884,
		0.1075782858,
		0.1025016378,
		0.0960887274,
		0.0884231585,
		0.0796048678,
		0.0697488238,
		0.0589835369,
		0.0474494125,
		0.0352970538,
		0.0226862316,
		0.0097989961
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 28)
	{
		double zero_points[28] = {
		-0.9964424976,
		-0.9813031654,
		-0.9542592806,
		-0.9156330264,
		-0.8658925226,
		-0.8056413709,
		-0.7356108780,
		-0.6566510940,
		-0.5697204718,
		-0.4758742250,
		-0.3762515161,
		-0.2720616276,
		-0.1645692821,
		-0.0550792899,
		0.0550792899,
		0.1645692821,
		0.2720616276,
		0.3762515161,
		0.4758742250,
		0.5697204718,
		0.6566510940,
		0.7356108780,
		0.8056413709,
		0.8658925226,
		0.9156330264,
		0.9542592806,
		0.9813031654,
		0.9964424976
		};
		//Gauss Weights
		double W[28] = {
		0.0091242826,
		0.0211321126,
		0.0329014278,
		0.0442729348,
		0.0551073457,
		0.0652729240,
		0.0746462142,
		0.0831134172,
		0.0905717444,
		0.0969306580,
		0.1021129676,
		0.1060557659,
		0.1087111923,
		0.1100470130,
		0.1100470130,
		0.1087111923,
		0.1060557659,
		0.1021129676,
		0.0969306580,
		0.0905717444,
		0.0831134172,
		0.0746462142,
		0.0652729240,
		0.0551073457,
		0.0442729348,
		0.0329014278,
		0.0211321126,
		0.0091242826
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 29)
	{
		double zero_points[29] = {
		-0.9966794423,
		-0.9825455053,
		-0.9572855958,
		-0.9211802330,
		-0.8746378049,
		-0.8181854876,
		-0.7524628517,
		-0.6782145376,
		-0.5962817971,
		-0.5075929551,
		-0.4131528882,
		-0.3140316379,
		-0.2113522862,
		-0.1062782301,
		0.0000000000,
		0.1062782301,
		0.2113522862,
		0.3140316379,
		0.4131528882,
		0.5075929551,
		0.5962817971,
		0.6782145376,
		0.7524628517,
		0.8181854876,
		0.8746378049,
		0.9211802330,
		0.9572855958,
		0.9825455053,
		0.9966794423
		};
		//Gauss Weights
		double W[29] = {
		0.0085169039,
		0.0197320851,
		0.0307404922,
		0.0414020625,
		0.0515948269,
		0.0612030907,
		0.0701179333,
		0.0782383271,
		0.0854722574,
		0.0917377571,
		0.0969638341,
		0.1010912738,
		0.1040733101,
		0.1058761551,
		0.1064793817,
		0.1058761551,
		0.1040733101,
		0.1010912738,
		0.0969638341,
		0.0917377571,
		0.0854722574,
		0.0782383271,
		0.0701179333,
		0.0612030907,
		0.0515948269,
		0.0414020625,
		0.0307404922,
		0.0197320851,
		0.0085169039
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 30)
	{
		double zero_points[30] = {
		-0.9968934841,
		-0.9836681233,
		-0.9600218650,
		-0.9262000474,
		-0.8825605358,
		-0.8295657624,
		-0.7677774321,
		-0.6978504948,
		-0.6205261830,
		-0.5366241481,
		-0.4470337695,
		-0.3527047255,
		-0.2546369262,
		-0.1538699136,
		-0.0514718426,
		0.0514718426,
		0.1538699136,
		0.2546369262,
		0.3527047255,
		0.4470337695,
		0.5366241481,
		0.6205261830,
		0.6978504948,
		0.7677774321,
		0.8295657624,
		0.8825605358,
		0.9262000474,
		0.9600218650,
		0.9836681233,
		0.9968934841
		};
		//Gauss Weights
		double W[30] = {
		0.0079681925,
		0.0184664683,
		0.0287847079,
		0.0387991926,
		0.0484026728,
		0.0574931562,
		0.0659742299,
		0.0737559747,
		0.0807558952,
		0.0868997872,
		0.0921225222,
		0.0963687372,
		0.0995934206,
		0.1017623897,
		0.1028526529,
		0.1028526529,
		0.1017623897,
		0.0995934206,
		0.0963687372,
		0.0921225222,
		0.0868997872,
		0.0807558952,
		0.0737559747,
		0.0659742299,
		0.0574931562,
		0.0484026728,
		0.0387991926,
		0.0287847079,
		0.0184664683,
		0.0079681925
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 31)
	{
		//Gauss Nodes in [-1.0, 1.0]
		double zero_points[31] = {
		-0.9970874818,
		-0.9846859097,
		-0.9625039251,
		-0.9307569979,
		-0.8897600299,
		-0.8399203201,
		-0.7817331484,
		-0.7157767846,
		-0.6427067229,
		-0.5632491614,
		-0.4781937820,
		-0.3883859016,
		-0.2947180700,
		-0.1981211993,
		-0.0995553122,
		0.0000000000,
		0.0995553122,
		0.1981211993,
		0.2947180700,
		0.3883859016,
		0.4781937820,
		0.5632491614,
		0.6427067229,
		0.7157767846,
		0.7817331484,
		0.8399203201,
		0.8897600299,
		0.9307569979,
		0.9625039251,
		0.9846859097,
		0.9970874818
		};
		//Gauss Weights
		double W[31] = {
		0.0074708316,
		0.0173186208,
		0.0270090192,
		0.0364322739,
		0.0454937075,
		0.0541030824,
		0.0621747866,
		0.0696285832,
		0.0763903866,
		0.0823929918,
		0.0875767406,
		0.0918901139,
		0.0952902429,
		0.0977433354,
		0.0992250112,
		0.0997205448,
		0.0992250112,
		0.0977433354,
		0.0952902429,
		0.0918901139,
		0.0875767406,
		0.0823929918,
		0.0763903866,
		0.0696285832,
		0.0621747866,
		0.0541030824,
		0.0454937075,
		0.0364322739,
		0.0270090192,
		0.0173186208,
		0.0074708316
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 32)
	{
		//Gauss Nodes in [-1.0, 1.0]
		double zero_points[32] = { -0.9972638618,
			-0.9856115115,
			-0.9647622556,
			-0.9349060759,
			-0.8963211558,
			-0.8493676137,
			-0.7944837960,
			-0.7321821187,
			-0.6630442669,
			-0.5877157572,
			-0.5068999089,
			-0.4213512761,
			-0.3318686023,
			-0.2392873623,
			-0.1444719616,
			-0.0483076657,
			0.0483076657,
			0.1444719616,
			0.2392873623,
			0.3318686023,
			0.4213512761,
			0.5068999089,
			0.5877157572,
			0.6630442669,
			0.7321821187,
			0.7944837960,
			0.8493676137,
			0.8963211558,
			0.9349060759,
			0.9647622556,
			0.9856115115,
			0.9972638618
		};
		//Gauss Weights
		double W[32] = {
		0.0070186100,
		0.0162743947,
		0.0253920653,
		0.0342738629,
		0.0428358980,
		0.0509980593,
		0.0586840935,
		0.0658222228,
		0.0723457941,
		0.0781938958,
		0.0833119242,
		0.0876520930,
		0.0911738787,
		0.0938443991,
		0.0956387201,
		0.0965400885,
		0.0965400885,
		0.0956387201,
		0.0938443991,
		0.0911738787,
		0.0876520930,
		0.0833119242,
		0.0781938958,
		0.0723457941,
		0.0658222228,
		0.0586840935,
		0.0509980593,
		0.0428358980,
		0.0342738629,
		0.0253920653,
		0.0162743947,
		0.0070186100
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 33)
	{
		//Gauss Nodes in [-1.0, 1.0]
		double zero_points[33] = {
	  -0.9974246942,
	  -0.9864557262,
	  -0.9668229097,
	  -0.9386943726,
	  -0.9023167677,
	  -0.8580096527,
	  -0.8061623563,
	  -0.7472304964,
	  -0.6817319600,
	  -0.6102423458,
	  -0.5333899048,
	  -0.4518500173,
	  -0.3663392577,
	  -0.2776090972,
	  -0.1864392988,
	  -0.0936310659,
	  0.0000000000,
	  0.0936310659,
	  0.1864392988,
	  0.2776090972,
	  0.3663392577,
	  0.4518500173,
	  0.5333899048,
	  0.6102423458,
	  0.6817319600,
	  0.7472304964,
	  0.8061623563,
	  0.8580096527,
	  0.9023167677,
	  0.9386943726,
	  0.9668229097,
	  0.9864557262,
	  0.9974246942
		};
		//Gauss Weights
		double W[33] = {
		0.0066062278,
		0.0153217015,
		0.0239155481,
		0.0323003586,
		0.0404015413,
		0.0481477428,
		0.0554708466,
		0.0623064825,
		0.0685945728,
		0.0742798548,
		0.0793123648,
		0.0836478761,
		0.0872482876,
		0.0900819587,
		0.0921239866,
		0.0933564261,
		0.0937684462,
		0.0933564261,
		0.0921239866,
		0.0900819587,
		0.0872482876,
		0.0836478761,
		0.0793123648,
		0.0742798548,
		0.0685945728,
		0.0623064825,
		0.0554708466,
		0.0481477428,
		0.0404015413,
		0.0323003586,
		0.0239155481,
		0.0153217015,
		0.0066062278
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
	else if (n == 34)
	{
		//Gauss Nodes in [-1.0, 1.0]
		double zero_points[34] = {
		  -0.9975717538,
		  -0.9872278164,
		  -0.9687082625,
		  -0.9421623974,
		  -0.9078096777,
		  -0.8659346383,
		  -0.8168842279,
		  -0.7610648766,
		  -0.6989391132,
		  -0.6310217271,
		  -0.5578755007,
		  -0.4801065452,
		  -0.3983592778,
		  -0.3133110813,
		  -0.2256666916,
		  -0.1361523573,
		  -0.0455098220,
		  0.0455098220,
		  0.1361523573,
		  0.2256666916,
		  0.3133110813,
		  0.3983592778,
		  0.4801065452,
		  0.5578755007,
		  0.6310217271,
		  0.6989391132,
		  0.7610648766,
		  0.8168842279,
		  0.8659346383,
		  0.9078096777,
		  0.9421623974,
		  0.9687082625,
		  0.9872278164,
		  0.9975717538
		};
		//Gauss Weights
		double W[34] = {
		0.0062291406,
		0.0144501627,
		0.0225637220,
		0.0304913806,
		0.0381665938,
		0.0455256115,
		0.0525074146,
		0.0590541358,
		0.0651115216,
		0.0706293758,
		0.0755619747,
		0.0798684443,
		0.0835130997,
		0.0864657397,
		0.0887018978,
		0.0902030444,
		0.0909567403,
		0.0909567403,
		0.0902030444,
		0.0887018978,
		0.0864657397,
		0.0835130997,
		0.0798684443,
		0.0755619747,
		0.0706293758,
		0.0651115216,
		0.0590541358,
		0.0525074146,
		0.0455256115,
		0.0381665938,
		0.0304913806,
		0.0225637220,
		0.0144501627,
		0.0062291406
		};
		for (int i = 0; i < n; i++)
		{
			x[i] = zero_points[i];
			A[i] = W[i];
		}

	}
}

void Gravity::SetDif_Lat_and_Dif_Lon1(const double Dif_Lat, const double Dif_Lon)
{
	Dif_Lat_ = Dif_Lat;
	Dif_Lon_ = Dif_Lon;
}

void Gravity::CalculatGravitation(std::vector<double>& sum, enum function func)
{
	SetSplit();
	/*参数*/
	int splited_s_size = s_size / split_;//卫星区域分块
	//int skip = 1;
	cudaMemcpyToSymbol(s_size_gpu, &splited_s_size, sizeof(int));
	cudaMemcpyToSymbol(g_size_gpu, &g_size, sizeof(int));
	cudaMemcpyToSymbol(p1_gpu, &p_1, sizeof(double));
	cudaMemcpyToSymbol(Dif_Lat_gpu, &Dif_Lat_, sizeof(double));
	cudaMemcpyToSymbol(Dif_Lon_gpu, &Dif_Lon_, sizeof(double));
	cudaMemcpyToSymbol(n_gpu, &n_, sizeof(int));
	cudaMemcpyToSymbol(x_gpu, x, sizeof(double) * n_);
	cudaMemcpyToSymbol(A_gpu, A, sizeof(double) * n_);

	/*计算最佳算力*/
	int minGridSize;// 完成最大算力的最小格网
	int blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, VZZ_GPU, 0, g_size * splited_s_size);
	int gridSize = (splited_s_size * g_size + blockSize - 1) / blockSize;//  

	/**************************************卫星区域分块*****************************/
	//参数
	PointXYZ* GroundPoints_GPU;
	PointXYZ* SatellitePoints_GPU;
	PointXYZ* Temp_SatellitePoints_;
	double* Results;
	double* Results_GPU;
	//分配内存
	Results = (double*)malloc(splited_s_size * sizeof(double));
	Temp_SatellitePoints_ = (PointXYZ*)malloc(splited_s_size * sizeof(PointXYZ));
	cudaMalloc((void**)&GroundPoints_GPU, g_size * sizeof(PointXYZ));
	cudaMalloc((void**)&SatellitePoints_GPU, splited_s_size * sizeof(PointXYZ));
	cudaMalloc((void**)&Results_GPU, splited_s_size * sizeof(double));
	cudaMemcpy(GroundPoints_GPU, GroundPoints_, g_size * sizeof(PointXYZ), cudaMemcpyHostToDevice);

	for (size_t i = 0; i < split_; i++)
	{
		//分块卫星区域
		for (size_t j = 0; j < splited_s_size; j++)
		{
			Temp_SatellitePoints_[j] = SatellitePoints_[j + i * splited_s_size];
		}
		cudaMemcpy(SatellitePoints_GPU, Temp_SatellitePoints_, splited_s_size * sizeof(PointXYZ), cudaMemcpyHostToDevice);
		cudaMemset(Results_GPU, 0, splited_s_size * sizeof(double));

		/*选择梯度*/
		switch (func)
		{
		case VX:
			VXX_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VY:
			VXY_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VZ:
			VXZ_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VXX:
			VXX_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VXY:
			VXY_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VXZ:
			VXZ_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VYY:
			VYY_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VYZ:
			VYZ_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		case VZZ:
			VZZ_GPU << <gridSize, blockSize, 0 >> > (GroundPoints_GPU, SatellitePoints_GPU, Results_GPU);
			break;
		default:
			break;
		}
		cudaMemcpy(Results, Results_GPU, sizeof(double) * splited_s_size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < splited_s_size; i++)
		{
			sum.push_back(Results[i]);
		}

	}

	/*释放显存*/
	cudaFree(GroundPoints_);
	cudaFree(GroundPoints_GPU);
	cudaFree(SatellitePoints_GPU);
	cudaFree(Temp_SatellitePoints_);
	cudaFree(Results);
	cudaFree(Results_GPU);
}

__global__ void VX_GPU(PointXYZ* GroundPoints, PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和
		int s = idx / s_size_gpu;

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;


		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vx_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__ void VY_GPU(PointXYZ* GroundPoints, PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和
		int s = idx / s_size_gpu;

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;


		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vy_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__ void VZ_GPU(PointXYZ* GroundPoints, PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和
		int s = idx / s_size_gpu;

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;


		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vz_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VXX_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和
		int s = idx / s_size_gpu;

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;


		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vxx_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VXY_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)
	{
		int step = idx % s_size_gpu;//将其直接求和

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;

		int s = idx / s_size_gpu;
		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vxy_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VXZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;

		//每次移动skip_gpu个点
		//for (int s = skip_gpu * (idx / s_size_gpu); s < skip_gpu + skip_gpu * (idx / s_size_gpu); s++)
		int s = idx / s_size_gpu;
		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vxz_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VYY_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;

		//每次移动skip_gpu个点
		//for (int s = skip_gpu * (idx / s_size_gpu); s < skip_gpu + skip_gpu * (idx / s_size_gpu); s++)
		int s = idx / s_size_gpu;
		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vyy_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VYZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = tid + bid * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;

		//每次移动skip_gpu个点
		//for (int s = skip_gpu * (idx / s_size_gpu); s < skip_gpu + skip_gpu * (idx / s_size_gpu); s++)
		int s = idx / s_size_gpu;
		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vyz_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);
				}
			}
		}
	}
}

__global__  void VZZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < s_size_gpu * g_size_gpu)//将450分解为45和10  / skip_gpu
	{
		int step = idx % s_size_gpu;//将其直接求和

		/*角度转化，角度转弧度，高度+地球半径*/
		double transfer = pi_gpu / 180;
		double Satellite_x_ = transfer * SatellitePoints[step].x;
		double Satellite_y_ = transfer * SatellitePoints[step].y;
		double Satellite_z_ = 6371000 + SatellitePoints[step].z;

		//每次移动skip_gpu个点
		//for (int s = skip_gpu * (idx / s_size_gpu); s < skip_gpu + skip_gpu * (idx / s_size_gpu); s++)
		int s = idx / s_size_gpu;
		/*万有引力常数*密度*/
		double p_gpu;
		if (GroundPoints[s].z >= 0)
			p_gpu = p1_gpu;
		else
			p_gpu = p2_gpu;
		double const_Gp = 66.7 * p_gpu;

		/*三重积分*/
		for (int i = 0; i < n_gpu; i++)
		{
			for (int j = 0; j < n_gpu; j++)
			{
				for (int k = 0; k < n_gpu; k++)
				{
					///*球面点,通过变换到标准区间[-1, 1]*/
					double suface_z = (GroundPoints[s].z * x_gpu[i] + GroundPoints[s].z) / 2 + 6371000;
					double suface_y = (Dif_Lon_gpu * x_gpu[j] + 2 * GroundPoints[s].y) * transfer / 2;
					double suface_x = (Dif_Lat_gpu * x_gpu[k] + 2 * GroundPoints[s].x) * transfer / 2;

					/*计算积分*/
					double temp = const_Gp * GroundPoints[s].z * Dif_Lat_gpu * transfer * Dif_Lon_gpu * transfer *
						A_gpu[i] * A_gpu[j] * A_gpu[k] * vzz_GPU(suface_x, suface_y, suface_z, Satellite_x_, Satellite_y_,
							Satellite_z_) / 8;//计算一个点的三重积分
					atomicAdd(&result_[step], temp);

				}
			}
		}

	}
}

__device__ double vx_GPU(double suface_x, double suface_y, double suface_z, double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = suface_z * k_fai * K / powf(L, 3);
	return f;
}

__device__ double vy_GPU(double suface_x, double suface_y, double suface_z, double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double k_lamba = cos(Satellite_y) * cos(suface_y) * sin(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = suface_z * k_lamba * K / (cos(Satellite_y) * powf(L, 3));
	return f;
}

__device__ double vz_GPU(double suface_x, double suface_y, double suface_z, double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (suface_z * cos_angle - Satellite_z) * K / powf(L, 3);
	return f;
}

__device__ double vxx_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (3 * (suface_z * suface_z) * (k_fai * k_fai) / powf(L, 5) - 1 / powf(L, 3)) * K;
	return f;
}

__device__ double vxy_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (3 * (suface_z * suface_z) * (k_fai * cos(suface_y) * sin(suface_x - Satellite_x)) / powf(L, 5)) * K;
	return f;
}

__device__ double vxz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (3 * suface_z * k_fai * (suface_z * cos_angle - Satellite_z) / powf(L, 5)) * K;
	return f;
}
__device__ double vyy_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = ((3 * suface_z * cos(suface_y) * sin(suface_x - Satellite_x) * suface_z * cos(suface_y) * sin(suface_x - Satellite_x) / powf(L, 5)) - 1 / powf(L, 3)) * K;
	return f;
}
__device__ double vyz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double k_fai = cos(Satellite_y) * sin(suface_y) - sin(Satellite_y) * cos(suface_y) * cos(suface_x - Satellite_x);
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (3 * suface_z * cos(suface_y) * sin(suface_x - Satellite_x) * (suface_z * cos_angle - Satellite_z) / powf(L, 5)) * K;
	return f;
}

__device__ double vzz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z)
{
	double f;
	double cos_angle = sin(suface_y) * sin(Satellite_y) + cos(suface_y) * cos(Satellite_y) * cos(Satellite_x - suface_x);
	double L = sqrt(suface_z * suface_z + Satellite_z * Satellite_z - 2 * suface_z * Satellite_z * cos_angle);
	double K = suface_z * suface_z * cos(suface_y);
	f = (3 * (suface_z * cos_angle - Satellite_z) * (suface_z * cos_angle - Satellite_z) / powf(L, 5) - 1 / powf(L, 3)) * K;
	return f;
}

void read_satellites(const std::string file_name, std::vector<PointXYZ>& data)
{
	/*读取文件*/
	std::ifstream file(file_name);
	if (file.bad()) {
		std::cout << "打开文件失败！" << std::endl;
		return;
	}
	PointXYZ temp;
	while (!file.eof())
	{
		file >> temp.x >> temp.y >> temp.z;
		data.push_back(temp);
	}
	file.close();

	std::cout << "读取观测点完成!" << std::endl;

}

void save_arc(const std::string save_file_name, std::vector<double>& I)
{
	std::ofstream outFile;
	//打开文件
	outFile.open(save_file_name, std::ios::out);
	int sum = 0;
	for (int i = 0; i < I.size(); i++)
	{
		outFile << I[i];//写入数据
		outFile << "\n";//写入数据
	}
	//关闭文件
	outFile.close();
}

void read_arc(const std::string file_name, std::vector<PointXYZ>& data, double& lat, double& lon)
{
	/*参数*/
	PointXYZ p;
	int i = 0;
	unsigned skip_y = 1;
	char szBuf[20];
	double step;
	double temp_x;
	unsigned row_, col_;
	int stop = 0;
	/*读取文件*/
	std::ifstream file(file_name);
	if (file.bad()) {
		std::cout << "打开文件失败！" << std::endl;
		return;
	}

	/*读取点坐标记录*/
	while (!file.eof())
	{
		if (i == 0)
		{
			file >> szBuf >> col_;
			i++;
			continue;
		}
		else if (i == 1)
		{
			file >> szBuf >> row_;
			i++;
			continue;

		}
		else if (i == 2)
		{
			file >> szBuf >> temp_x;
			p.x = temp_x;
			i++;
			continue;
		}
		else if (i == 3)
		{
			file >> szBuf >> p.y;
			i++;
			continue;
		}
		else if (i == 4)
		{
			file >> szBuf >> step;
			lat = step;/*格网维度距离*/
			lon = step;/*格网维度距离*/
			i++;
			continue;
		}
		else if (i == 5)
		{
			double skip;
			file >> szBuf >> skip;
			i++;
			continue;
		}
		else
		{
			file >> p.z;
			data.push_back(p);
			if (skip_y == (row_ * col_) || file.peek() == EOF)
			{
				break;
			}
			p.x += step;
			if (skip_y % col_ == 0)
			{
				stop++;
				p.y += step;
				p.x = temp_x;
			}
			skip_y++;
		}
	}
	file.close();

	std::cout << "读取地形点完成!" << std::endl;
}

