// 引用头文件 
// head file
#include "libTac3D.hpp"

// 用于存储三维形貌、三维变形场、三维分布力、三维合力、三维合力矩数据的矩阵
// Mat for storing point cloud, displacement field, distributed force, resultant force, and resultant moment
cv::Mat P, D, F, Fr, Mr;

// 帧序号
// frame index
int frameIndex;

// 时间戳
// timestamp
double sendTimestamp, recvTimestamp;

// 传感器SN
// Serial Number of the Tac3D sensor
std::string SN;  

// 编写用于接收数据的回调函数
// Write a callback function to receive data
void Tac3DRecvCallback(Tac3D::Frame &frame, void *param)
{
    cv::Mat *tempMat;

    // 初始化Sensor类时传入的自定义参数
    // Custom parameter passed in when initializing the Sensor object
    float *testParam = (float*)param;

    // 获得传感器SN码，可用于区分触觉信息来源于哪个触觉传感器
    // get the SN code, which can be used to distinguish which Tac3D sensor the tactile information comes from
    SN = frame.SN; 

    // 获得帧序号
    // get the frame index
    frameIndex = frame.index;  

    // 获得时间戳
    // get the timestamp
    sendTimestamp = frame.sendTimestamp;
    recvTimestamp = frame.recvTimestamp;
    
    // 使用frame.get函数通过数据名称"3D_Positions"获得cv::Mat类型的三维形貌的数据指针
    // 矩阵的3列分别为x,y,z方向的分量
    // 矩阵的每行对应一个测量点
    // Use the frame.get function to obtain the data pointer of the 3D shape of the cv::Mat type through the data name "3D_Positions"
    // The three columns of the matrix are the components in the x, y, and z directions, respectively
    // Each row of the matrix corresponds to a sensing point
    tempMat = frame.get<cv::Mat>("3D_Positions");

    // 当Tac3DRecvCallback函数返回时，frame的内存可能会被回收。如果需要在其他位置使用获取的数据，需要用copyTo函数将数据复制到其他位置
    // When the Tac3DRecvCallback function returns, the frame's memory may be reclaimed. If you need to use the acquired data elsewhere, you need to use the "copyTo" function to copy the data to another place.
    tempMat->copyTo(P);

    // 使用frame.get函数通过数据名称"3D_Displacements"获得cv::Mat类型的三维变形场的数据指针
    // 矩阵的3列分别为x,y,z方向的分量
    // 矩阵的每行对应一个测量点
    // Use the frame.get function to obtain the data pointer of the displacement field of the cv::Mat type through the data name "3D_Displacements"
    // The three columns of the matrix are the components in the x, y, and z directions, respectively
    // Each row of the matrix corresponds to a sensing point
    tempMat = frame.get<cv::Mat>("3D_Displacements");
    tempMat->copyTo(D);

    // 使用frame.get函数通过数据名称"3D_Forces"获得cv::Mat类型的三维分布力的数据指针
    // 矩阵的3列分别为x,y,z方向的分量
    // 矩阵的每行对应一个测量点
    // Use the frame.get function to obtain the data pointer of the distributed force of the cv::Mat type through the data name "3D_Forces"
    // The three columns of the matrix are the components in the x, y, and z directions, respectively
    // Each row of the matrix corresponds to a sensing point
    tempMat = frame.get<cv::Mat>("3D_Forces");
    tempMat->copyTo(F);

    // 使用frame.get函数通过数据名称"3D_ResultantForce"获得cv::Mat类型的三维合力的数据指针
    // 矩阵的3列分别为x,y,z方向的分量
    // Use the frame.get function to obtain the data pointer of the resultant force of the cv::Mat type through the data name "3D_ResultantForce"
    // The three columns of the matrix are the components in the x, y, and z directions, respectively
    tempMat = frame.get<cv::Mat>("3D_ResultantForce");
    tempMat->copyTo(Fr);

    // 使用frame.get函数通过数据名称"3D_ResultantMoment"获得cv::Mat类型的三维合力的数据指针
    // 矩阵的3列分别为x,y,z方向的分量
    // Use the frame.get function to obtain the data pointer of the resultant moment of the cv::Mat type through the data name "3D_ResultantMoment"
    // The three columns of the matrix are the components in the x, y, and z directions, respectively
    tempMat = frame.get<cv::Mat>("3D_ResultantMoment");
    tempMat->copyTo(Mr);
}


int main(int argc,char **argv)
{
    std::cout << "libTac3D version is : " << LIBTAC3D_VERSION << std::endl;

    float testParam = 100.0;

    // 创建Sensor实例，设置回调函数为上面写好的Tac3DRecvCallback，设置UDP接收端口为9988
    // 每次接收到数据帧时，会自动调用Tac3DRecvCallback函数
    // Create a Sensor object, set the callback function to Tac3DRecvCallback, and set the UDP receive port to 9988
    // The Tac3DRecvCallback function will be automatically called every time a data frame is received
    Tac3D::Sensor tac3d(Tac3DRecvCallback, 9988, &testParam); 

    // 等待Tac3D传感器启动并传来数据
    // Wait for the Tac3D sensor to start and send data
    tac3d.waitForFrame();

    // 5s
    usleep(1000*1000*5);


    // 发送一次校准信号（应确保校准时传感器未与任何物体接触）
    // Send a calibration signal to reset zero point (please ensure that the sensor is not in contact with any object during calibration)
    tac3d.calibrate(SN);

    // 5s
    usleep(1000*1000*5);
}
