/****************************************************************************
* @file    VZZ_GPU.cu
* @author  cui hao
* @version V3.5.0
* @date    28-December-2021
* @brief   CPU��GPU����������ֱ�ݶ�
******************************************************************************
* @attention ��Ҫʹ�ø�˹-���õ·���
*
******************************************************************************/

#include<stdio.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<vector>

#include"cuda_runtime.h"
#include<cuda_runtime.h>//CUDA��ʱ��
#include <device_launch_parameters.h>//threadIdx.x
#include "Gravity_GPU.cuh"


int main()
{
	std::string file_name = "data/test_time.asc"; /*������ļ���*/
	std::string satellite_file_name = "data/test_time_satellite.asc"; /*������ļ���*/
	std::string save_file_name = "data/test_result11.txt"; /*�����ļ���*/
	double density = 2.67;				/*�ܶȣ�ˮ���ܶ�Ϊ0*/
	unsigned nodes = 5;					/*�ڵ���*/
	enum function func = VZZ;			/*��������ݶ�*/

	//--------------------------------------------��������----------------------------------------------------
	std::vector<double> I;
	std::vector<PointXYZ> Satellite;
	std::vector<PointXYZ> surfaces;
	double Lat = 0;	/*����ά�Ⱦ���*/
	double Lon = 0;	/*�������Ⱦ���*/
	/*��ȡ�ļ�*/
	read_arc(file_name, surfaces, Lat, Lon);
	/*��������*/
	read_satellites(satellite_file_name, Satellite);

	//--------------------------------------------���������ݶ�----------------------------------------------------
	clock_t begin = clock();
	Gravity gra;
	gra.SetGroundPoints(surfaces);
	gra.SetSatellitePoints(Satellite);
	gra.SetDensity(density);
	gra.Setn(nodes);
	gra.SetDif_Lat_and_Dif_Lon1(Lat, Lon);
	gra.CalculatGravitation(I, func);
	/*��ʱ����*/
	clock_t end = clock() - begin;
	std::printf("ʱ�䣺%d(s)", end / CLOCKS_PER_SEC);

	//-----------------------------------------------��������----------------------------------------------------
	save_arc(save_file_name, I);
	return 0;
}




