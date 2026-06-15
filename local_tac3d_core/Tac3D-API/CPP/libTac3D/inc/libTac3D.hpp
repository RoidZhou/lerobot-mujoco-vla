#ifndef _LIBTAC3D_HPP
#define _LIBTAC3D_HPP

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <map>
#include <thread>

#define TAC3D_NETWORKRECEIVE_BUFFSIZE 1024000
#define TAC3D_NETWORKRECEIVE_UDPPACKETSIZE 1400  // 这个值必须和NETWORKTRANSPORT_UDPPACKETSIZE一致!!
#define TAC3D_NETWORKRECEIVE_RECVPOOLSIZE 30
#define TAC3D_NETWORKRECEIVE_RECVTIMEOUT 1.0

#define LIBTAC3D_VERSION "3.3.0"

#ifdef _WIN32
#include <windows.h>
#include <time.h>

void usleep(DWORD t_us);

#define socklen_t int
#else
#include <arpa/inet.h>   //htonl  htons 所需的头文件
#include <unistd.h>
#include <sys/time.h>
#endif //_WIN32

namespace Tac3D
{

class Timer
{
public:
    struct timeval startTime;
    struct timeval checkTime;
    struct timeval currentTime;

    double Start();
    double GetTime();
    double Check();
};

typedef struct
{
    int dataLength;
    char *data;
    socklen_t addr_len;
    sockaddr_in addr;
}SockUDPFrame_t;

class UDPSock
{
public:
    UDPSock(void);
    void Start(int port = 8083, bool isServer = false);
    void SetCallback(void (*callback)(SockUDPFrame_t *frame, void *param), void *param);
    ~UDPSock();
    bool Send(SockUDPFrame_t *frame);
    bool SetAddr(SockUDPFrame_t *frame, const char* ip, int port);

private:
    void ServSockBind(int port);
    void Receive();
    int udp_sock;
    sockaddr_in udp_addr;
    bool running;
    void (*recvCallback)(SockUDPFrame_t *frame, void *param);
    void *recvCallbackParam;
};

struct NetworkReceiveBuffer
{
    int bufferIndex;
    bool isFree;
    double loaclTimestamp;
    uint32_t serialNum;

    uint16_t pktNum;
    uint16_t pktCnt;

    int headLen;
    int dataLen;
    char headBuffer[TAC3D_NETWORKRECEIVE_UDPPACKETSIZE];
    char dataBuffer[TAC3D_NETWORKRECEIVE_BUFFSIZE];
};

class Frame
{
private:
    std::map<std::string, void*> fieldCache;

public:
    uint32_t index = 0;
    std::string SN;
    std::string message;
    double sendTimestamp;
    double recvTimestamp;
    
    void _addFrameField(std::string fieldName, void* ptr);

    template <typename T>
    T* get(std::string fieldName, bool notFoundWarning = true)
    {
        if (Frame::fieldCache.find(fieldName) != Frame::fieldCache.end())
        {
            return (T*)Frame::fieldCache[fieldName];
        }
        else if (notFoundWarning)
        {
            std::cout << "frame field not found : " << fieldName << " (get).";
        }
        return NULL;
    }
    void dumpField();
};


class Sensor
{
private:
    UDPSock UDP;
    int port;
    std::map<std::string, sockaddr_in> fromAddrMap;
    std::map<std::string, int> typeDict;
    Frame recvFrame;
    bool _isReady;
    void (*_recvCallback)(Frame &frame, void *param) = NULL;
    void *_callbackParam;
public:
    Timer timer;
    NetworkReceiveBuffer *bufferPool = NULL;
    NetworkReceiveBuffer* _GetFreeBuffer();
    NetworkReceiveBuffer* _GetBufferBySerialNum(uint32_t serialNum);
    void _process(NetworkReceiveBuffer &buffer, sockaddr_in &fromAddr);
    ~Sensor();

    
    Sensor(void (*recvCallback)(Frame &frame, void *param), int port, void* callbackParam = NULL);

    // send calibration signal to Tac3D-Core
    void calibrate(std::string SN);

    // send quit signal to Tac3D-Core
    void quitSensor(std::string SN);

    // block until frames received
    void waitForFrame(std::string SN = "");
};

}

#endif

