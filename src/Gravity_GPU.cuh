#pragma once
#include<vector>
/*thrust��*/
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>//CUDA��ʱ��
#include <fstream>                  /*�ļ�����*/


/*�����������*/
enum function
{
	VX, VY, VZ, VXX, VXY, VXZ, VYY, VYZ, VZZ
};

/*��������*/
struct PointXYZ
{
	double x, y, z;
};

class Gravity
{
private:
	double p_1;								/*�ܶ�*/
	double p_2;								/*ˮ���ܶ�*/
	size_t g_size;							/*���������*/
	size_t s_size;							/*���ǵ�����*/
	unsigned n_;							/*�ڵ���*/
	size_t g_row_;							/*��*/
	size_t g_col_;							/*��*/
	double Dif_Lat_;						/*���Ȼ���*/
	double Dif_Lon_;						/*ά�Ȼ���*/
	double x[34], A[34];					/*ϵ��*/
	unsigned split_;						/*�ֿ�*/
	PointXYZ* SatellitePoints_;				/*���ǵ�*/
	PointXYZ* GroundPoints_;				/*���ǵ�*/

	/*����ֿ�*/
	void SetSplit();

	/*�Ƿ�Ϊ����*/
	bool is_Integer(double input);
public:
	/*������������*/
	void SetGroundPoints(std::vector<PointXYZ> GroundPoints);

	/*������������*/
	void SetSatellitePoints(std::vector<PointXYZ> SatellitePoints);

	///*���������к���*/
	//void SetRowandCol(const unsigned row__, const unsigned col__);

	/*�����ܶ�*/
	void SetDensity(const double p);

	/*����ڵ���*/
	void Setn(const unsigned n);

	/*����ά�Ȳ�;��Ȳ�*/
	void SetDif_Lat_and_Dif_Lon1(const double Dif_Lat, const double Dif_Lon);

	/*��������*/
	void CalculatGravitation(std::vector<double>& sum, enum function func);
};
__global__  void VX_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VY_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VXX_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VXY_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VXZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VYY_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VYZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__global__  void VZZ_GPU(PointXYZ* GroundPoints,
	PointXYZ* SatellitePoints, double* result_);

__device__ double vx_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vy_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vxx_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vxy_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vxz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vyy_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vyz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

__device__ double vzz_GPU(double suface_x, double suface_y, double suface_z,
	double Satellite_x, double Satellite_y, double Satellite_z);

void read_arc(const std::string file_name, std::vector<PointXYZ>& data, double& lat, double& lon);

void read_satellites(const std::string file_name, std::vector<PointXYZ>& data);

void save_arc(const std::string save_file_name, std::vector<double>& I);