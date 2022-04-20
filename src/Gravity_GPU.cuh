#pragma once
#include<vector>
/*thrust库*/
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>//CUDA记时器
#include <fstream>                  /*文件操作*/


/*方便输入参数*/
enum function
{
	VX, VY, VZ, VXX, VXY, VXZ, VYY, VYZ, VZZ
};

/*数据类型*/
struct PointXYZ
{
	double x, y, z;
};

class Gravity
{
private:
	double p_1;								/*密度*/
	double p_2;								/*水下密度*/
	size_t g_size;							/*地面点数量*/
	size_t s_size;							/*卫星点数量*/
	unsigned n_;							/*节点数*/
	size_t g_row_;							/*行*/
	size_t g_col_;							/*列*/
	double Dif_Lat_;						/*经度划分*/
	double Dif_Lon_;						/*维度划分*/
	double x[34], A[34];					/*系数*/
	unsigned split_;						/*分块*/
	PointXYZ* SatellitePoints_;				/*卫星点*/
	PointXYZ* GroundPoints_;				/*卫星点*/

	/*输入分块*/
	void SetSplit();

	/*是否为整数*/
	bool is_Integer(double input);
public:
	/*输入地面点数据*/
	void SetGroundPoints(std::vector<PointXYZ> GroundPoints);

	/*输入卫星数据*/
	void SetSatellitePoints(std::vector<PointXYZ> SatellitePoints);

	///*输入地面点行和列*/
	//void SetRowandCol(const unsigned row__, const unsigned col__);

	/*输入密度*/
	void SetDensity(const double p);

	/*输入节点数*/
	void Setn(const unsigned n);

	/*输入维度差和经度差*/
	void SetDif_Lat_and_Dif_Lon1(const double Dif_Lat, const double Dif_Lon);

	/*计算引力*/
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