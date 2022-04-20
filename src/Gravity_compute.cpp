/****************************************************************************
* @file    VZZ_GPU.cu
* @author  cui hao
* @version V3.5.0
* @date    28-December-2021
* @brief   CPU和GPU计算重力垂直梯度
******************************************************************************
* @attention 主要使用高斯-勒让德方法
*
******************************************************************************/

#include<stdio.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<vector>

#include"cuda_runtime.h"
#include<cuda_runtime.h>//CUDA记时器
#include <device_launch_parameters.h>//threadIdx.x
#include "Gravity_GPU.cuh"


int main()
{
	std::string file_name = "data/test_time.asc"; /*地面点文件名*/
	std::string satellite_file_name = "data/test_time_satellite.asc"; /*地面点文件名*/
	std::string save_file_name = "data/test_result11.txt"; /*保存文件名*/
	double density = 2.67;				/*密度，水下密度为0*/
	unsigned nodes = 5;					/*节点数*/
	enum function func = VZZ;			/*计算何种梯度*/

	//--------------------------------------------输入数据----------------------------------------------------
	std::vector<double> I;
	std::vector<PointXYZ> Satellite;
	std::vector<PointXYZ> surfaces;
	double Lat = 0;	/*格网维度距离*/
	double Lon = 0;	/*格网精度距离*/
	/*读取文件*/
	read_arc(file_name, surfaces, Lat, Lon);
	/*卫星数据*/
	read_satellites(satellite_file_name, Satellite);

	//--------------------------------------------计算重力梯度----------------------------------------------------
	clock_t begin = clock();
	Gravity gra;
	gra.SetGroundPoints(surfaces);
	gra.SetSatellitePoints(Satellite);
	gra.SetDensity(density);
	gra.Setn(nodes);
	gra.SetDif_Lat_and_Dif_Lon1(Lat, Lon);
	gra.CalculatGravitation(I, func);
	/*计时结束*/
	clock_t end = clock() - begin;
	std::printf("时间：%d(s)", end / CLOCKS_PER_SEC);

	//-----------------------------------------------保存数据----------------------------------------------------
	save_arc(save_file_name, I);
	return 0;
}




