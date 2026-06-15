#include "libTac3D.hpp"


#define NETWORKRECEIVE_TYPE_MAT 0
#define NETWORKRECEIVE_TYPE_I32 1
#define NETWORKRECEIVE_TYPE_F64 2
#define NETWORKRECEIVE_TYPE_IMG 3


#ifdef _WIN32

void usleep(DWORD t_us)
{
    Sleep(t_us / 1000);
}
#endif

namespace Tac3D
{


#ifdef _WIN32

static int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    ULARGE_INTEGER uliTime;
    uliTime.LowPart = ft.dwLowDateTime;
    uliTime.HighPart = ft.dwHighDateTime;
    uint64_t ullTime = uliTime.QuadPart;

    static const uint64_t EPOCH_DIFFERENCE = 116444736000000000ULL;
    ullTime -= EPOCH_DIFFERENCE;
    ullTime /= 10;

    tv->tv_sec = ullTime / 1000000;
    tv->tv_usec = ullTime % 1000000;

    return 0;
}

#endif

double Timer::Start()
{
    gettimeofday(&(Timer::startTime), NULL);
    Timer::Check();
    return 0.0;
}

double Timer::GetTime()
{
    gettimeofday(&(Timer::currentTime), NULL);
    return (double)(Timer::currentTime.tv_sec - Timer::startTime.tv_sec) + \
        (double)(Timer::currentTime.tv_usec - Timer::startTime.tv_usec) / 1000000;
}

double Timer::Check()
{
    double interval;
    gettimeofday(&(Timer::currentTime), NULL);
    interval = (double)(Timer::currentTime.tv_sec - Timer::checkTime.tv_sec) + \
        (double)(Timer::currentTime.tv_usec - Timer::checkTime.tv_usec) / 1000000;
    Timer::checkTime = Timer::currentTime;
    return interval;
}

static void EmptyCallback(SockUDPFrame_t *frame, void *params)
{}

UDPSock::UDPSock()
{
    UDPSock::recvCallback = EmptyCallback;
    UDPSock::recvCallbackParam = NULL;
}
void UDPSock::SetCallback(void (*callback)(SockUDPFrame_t *frame, void *param), void *param)
{
    UDPSock::recvCallback = callback;
    UDPSock::recvCallbackParam = param;
}

void UDPSock::Start(int port, bool isServer)
{
    try{
        
#ifdef _WIN32
        WORD wVersionRequested;
        WSADATA wsaData;
        int err;
        wVersionRequested = MAKEWORD( 1, 1 );
        err = WSAStartup( wVersionRequested, &wsaData );
        if ( err != 0 )
        {
            throw "socket error";
        }
#endif

        udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); //UDP
	    if(udp_sock == -1)
	    {
            throw "socket error";
	    };
    }catch(const char *msg){
        std::cerr<<msg<<std::endl;
        exit(-1);
    }
    if (isServer)
    {
        ServSockBind(port);
    }
    UDPSock::running = true;
    std::thread recvThread(&UDPSock::Receive, this);
    recvThread.detach();
}

UDPSock::~UDPSock()
{
}

void UDPSock::ServSockBind(int port)
{
    memset(&udp_addr, 0, sizeof(udp_addr));
	udp_addr.sin_family = AF_INET;
	udp_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    int sockPort = port;
    try
    {
        if (sockPort<=0||sockPort>65535)
        {
            throw "invalid port";
        }
        udp_addr.sin_port = htons(sockPort);
    }
    catch(const char *msg){
        std::cerr<<msg<<std::endl;
        exit(-1);
    }
    bind(udp_sock, (struct sockaddr*)&udp_addr, sizeof(udp_addr));
}

bool UDPSock::Send(SockUDPFrame_t *frame)
{
    int sendLen;
    sendLen = sendto(udp_sock, frame->data, frame->dataLength, 0, (sockaddr *)&frame->addr, frame->addr_len);
    if (sendLen != frame->dataLength)
    {
        std::cerr<<"UDP send failed!"<<std::endl; 
        return false;
    }
    else
    {
        return true;
    }
}

void UDPSock::Receive()
{
    SockUDPFrame_t frame;
    char data[65535];
    frame.data = data;
    frame.addr_len = sizeof(sockaddr_in);

    while (UDPSock::running)
    {
        
        frame.dataLength = recvfrom(udp_sock, frame.data, 65535, 0, (sockaddr *)&frame.addr, &frame.addr_len);
        
        if (frame.dataLength < 0)
        {
            usleep(1000);
#ifdef _WIN32
            printf("recvfrom failed:%d\n", WSAGetLastError());
#else
            printf("recvfrom failed\n");
#endif
        }
        else
        {
            UDPSock::recvCallback(&frame, UDPSock::recvCallbackParam);
        }
    }
}

bool UDPSock::SetAddr(SockUDPFrame_t *frame, const char* ip, int port)
{
    sockaddr_in temp_udp_addr;
    temp_udp_addr.sin_family = AF_INET;
    try
    {
        // server port and ip
        int sockPort = port;
        temp_udp_addr.sin_addr.s_addr = inet_addr(ip);
        if (sockPort<=0||sockPort>65535)
        {
            throw "invalid remote address";
        }
        temp_udp_addr.sin_port = htons(sockPort);

    }
    catch(const char *msg)
    {
        std::cerr<<msg<<std::endl;
        exit(-1);
    }

    frame->addr = temp_udp_addr;
    frame->addr_len = sizeof(udp_addr);
    return true;
}

static void NetworkRecvCallback(SockUDPFrame_t *frame, void *param)
{
    Sensor *psensor = (Sensor *)param;
    NetworkReceiveBuffer *tempBuffer;
    uint32_t serialNum;
    uint16_t curPktCnt;
    char *dataPtr = (char*)frame->data;

    memcpy(&serialNum, dataPtr, 4);
    dataPtr += 4;
    tempBuffer = psensor->_GetBufferBySerialNum(serialNum);
    if (tempBuffer == NULL)
    {
        tempBuffer = psensor->_GetFreeBuffer();
        if (tempBuffer == NULL)
        {
            return;
        }
        
        memcpy(&tempBuffer->pktNum, dataPtr, 2);
        if ((tempBuffer->pktNum + 1) * TAC3D_NETWORKRECEIVE_UDPPACKETSIZE > TAC3D_NETWORKRECEIVE_BUFFSIZE)
        {
            return;
        }
        tempBuffer->isFree = false;
        tempBuffer->dataLen = 0;
        tempBuffer->headLen = 0;
        tempBuffer->serialNum = serialNum;
        tempBuffer->pktCnt = 0;
        memcpy(&tempBuffer->pktNum, dataPtr, 2);
    }
    dataPtr += 2;
    memcpy(&curPktCnt, dataPtr, 2);
    dataPtr += 2;

    tempBuffer->loaclTimestamp = psensor->timer.GetTime();

    if (curPktCnt == 0) // head
    {
        tempBuffer->headLen = frame->dataLength - 8;
        memcpy(tempBuffer->headBuffer, dataPtr, tempBuffer->headLen);
        tempBuffer->headBuffer[tempBuffer->headLen] = '\0';
    }
    else
    {
        int offset, pktDataLen;
        offset = (curPktCnt - 1) * (TAC3D_NETWORKRECEIVE_UDPPACKETSIZE - 8);
        pktDataLen = frame->dataLength - 8;
        memcpy(tempBuffer->dataBuffer+offset, dataPtr, pktDataLen);
        tempBuffer->dataLen += pktDataLen;
    }
    tempBuffer->pktCnt++;
    if (tempBuffer->pktCnt == tempBuffer->pktNum + 1)
    {
        psensor->_process(*tempBuffer, frame->addr);
        tempBuffer->isFree = true;
    }
}

void Frame::_addFrameField(std::string fieldName, void* ptr)
{
    Frame::fieldCache[fieldName] = ptr;
}

void Frame::dumpField()
{
    std::map<std::string, void*>::iterator iter;

    std::cout << "cache fields:\n";
    for (iter = Frame::fieldCache.begin(); iter != Frame::fieldCache.end(); iter++)
    {
        std::cout << iter->first << " : " << iter->second << "\n";
    }
    std::cout << std::endl;
}

NetworkReceiveBuffer* Sensor::_GetFreeBuffer()
{
    NetworkReceiveBuffer *tempBuffer;
    double currTime = Sensor::timer.GetTime();
    for (int i = 0; i < TAC3D_NETWORKRECEIVE_RECVPOOLSIZE; i++)
    {
        tempBuffer = &(Sensor::bufferPool[i]);
        if (tempBuffer->isFree)
        {
            return tempBuffer;
        }
        else if (currTime - tempBuffer->loaclTimestamp > TAC3D_NETWORKRECEIVE_RECVTIMEOUT)
        {
            tempBuffer->isFree = true;
            return tempBuffer;
        }
    }
    return NULL;
}

NetworkReceiveBuffer* Sensor::_GetBufferBySerialNum(uint32_t serialNum)
{
    NetworkReceiveBuffer *tempBuffer;
    for (int i = 0; i < TAC3D_NETWORKRECEIVE_RECVPOOLSIZE; i++)
    {
        tempBuffer = &(Sensor::bufferPool[i]);
        if (!tempBuffer->isFree && tempBuffer->serialNum == serialNum)
        {
            return tempBuffer;
        }
    }
    return NULL;
}

Sensor::Sensor(void (*recvCallback)(Frame &frame, void *param), int port, void* callbackParam)
{
    Sensor::port = port;
    Sensor::timer.Start();
    Sensor::_recvCallback = recvCallback;
    Sensor::_callbackParam = callbackParam;
    Sensor::_isReady = false;
    Sensor::bufferPool = new NetworkReceiveBuffer[TAC3D_NETWORKRECEIVE_RECVPOOLSIZE];
    for (int i = 0; i < TAC3D_NETWORKRECEIVE_RECVPOOLSIZE; i++)
    {
        Sensor::bufferPool[i].isFree = true;
        Sensor::bufferPool[i].bufferIndex = i;
    }

    Sensor::typeDict["mat"] = NETWORKRECEIVE_TYPE_MAT;
    Sensor::typeDict["i32"] = NETWORKRECEIVE_TYPE_I32;
    Sensor::typeDict["f64"] = NETWORKRECEIVE_TYPE_F64;
    Sensor::typeDict["img"] = NETWORKRECEIVE_TYPE_IMG;

    Sensor::UDP.SetCallback(NetworkRecvCallback, this);
    Sensor::UDP.Start(Sensor::port, true);
}

Sensor::~Sensor()
{
    if (Sensor::bufferPool != NULL)
    {
        delete Sensor::bufferPool;
    }
}

void Sensor::waitForFrame(std::string SN)
{    
    std::cout << "Waiting for Tac3D sensor " << SN << "..." << std::endl;
    if (SN == std::string(""))
    {
        Sensor::_isReady = false;
        while (!Sensor::_isReady)
        {
            usleep(10000);
        }
    }
    else
    {
        if (Sensor::fromAddrMap.find(SN) != Sensor::fromAddrMap.end())
        {
            Sensor::fromAddrMap.erase(SN);
        }
        while (Sensor::fromAddrMap.find(SN) == Sensor::fromAddrMap.end())
        {
            usleep(10000);
        }
    }
    std::cout << "Tac3D sensor connected." << std::endl;
}

void Sensor::_process(NetworkReceiveBuffer &buffer, sockaddr_in &fromAddr)
{
    
    cv::Mat *tempMat;
    int32_t *tempI32;
    double *tempF64;
    std::vector<unsigned char> tempBuffer;
    YAML::Node dataNode;
    std::string fieldName;
    std::string fieldType;
    int fieldOffet;
    int fieldLength;

    YAML::Node headInfo = YAML::Load((const char*)buffer.headBuffer);
    YAML::Node dataInfo = headInfo["data"];
    
    Sensor::recvFrame.SN = headInfo["SN"].as<std::string>();
    Sensor::recvFrame.index = headInfo["index"].as<int>();
    Sensor::recvFrame.sendTimestamp = headInfo["timestamp"].as<double>();
    Sensor::recvFrame.recvTimestamp = Sensor::timer.GetTime();

    if (headInfo["msg"].IsDefined())  // 兼容3.3.0之前的版本
    {
        Sensor::recvFrame.message = headInfo["msg"].as<std::string>();
        if (Sensor::recvFrame.message.length() > 0)
        {
            std::cout << "[" << Sensor::recvFrame.SN << "] " << Sensor::recvFrame.message << std::endl;
        }
    }
    else
    {
        Sensor::recvFrame.message = "";
    }

    Sensor::fromAddrMap[Sensor::recvFrame.SN] = fromAddr;

    for (size_t i = 0; i < dataInfo.size(); i++)
    {
        dataNode = dataInfo[i];
        fieldName = dataNode["name"].as<std::string>();
        fieldType = dataNode["type"].as<std::string>();
        fieldOffet = dataNode["offset"].as<int>();
        fieldLength = dataNode["length"].as<int>();

        switch (Sensor::typeDict[fieldType])
        {
        case NETWORKRECEIVE_TYPE_MAT:
            tempMat = Sensor::recvFrame.get<cv::Mat>(fieldName, false);
            if (tempMat == NULL)
            {
                tempMat = new cv::Mat;
                Sensor::recvFrame._addFrameField(fieldName, tempMat);
            }
            tempMat->create(dataNode["height"].as<int>(), dataNode["width"].as<int>(), CV_64FC1);
            memcpy(tempMat->data, buffer.dataBuffer+fieldOffet, fieldLength);
            break;
        case NETWORKRECEIVE_TYPE_I32:
            tempI32 = Sensor::recvFrame.get<int32_t>(fieldName, false);
            if (tempI32 == NULL)
            {
                tempI32 = new int32_t;
                Sensor::recvFrame._addFrameField(fieldName, tempI32);
            }
            memcpy(tempI32, buffer.dataBuffer+fieldOffet, fieldLength);
            break;
        case NETWORKRECEIVE_TYPE_F64:
            tempF64 = Sensor::recvFrame.get<double>(fieldName, false);
            if (tempF64 == NULL)
            {
                tempF64 = new double;
                Sensor::recvFrame._addFrameField(fieldName, tempF64);
            }
            memcpy(tempF64, buffer.dataBuffer+fieldOffet, fieldLength);
            break;
        case NETWORKRECEIVE_TYPE_IMG:
            tempMat = Sensor::recvFrame.get<cv::Mat>(fieldName, false);
            if (tempMat == NULL)
            {
                tempMat = new cv::Mat;
                Sensor::recvFrame._addFrameField(fieldName, tempMat);
            }
            tempBuffer.insert(tempBuffer.end(), buffer.dataBuffer+fieldOffet, buffer.dataBuffer+fieldOffet+fieldLength);
            cv::imdecode(tempBuffer, cv::IMREAD_ANYCOLOR, tempMat);
            break;
        default:
            std::cout << "No such field type named " << fieldType << " (recv)." << std::endl;
            break;
        }
    }
    
    tempF64 = Sensor::recvFrame.get<double>("InitializeProgress", false);
    if (tempF64 != NULL)
    {
        if (*tempF64 != 100) 
        {
            return;
        }
    }
    Sensor::_isReady = true;
    Sensor::_recvCallback(Sensor::recvFrame, Sensor::_callbackParam);
}

void Sensor::calibrate(std::string SN)
{
    SockUDPFrame_t sendFrame;
    char sendData[2] = {'$', 'C'};
    if (Sensor::fromAddrMap.find(SN) == Sensor::fromAddrMap.end())
    {
        std::cout << "Calibtation failed! (sensor " << SN << " is not connected)" << std::endl;
    }
    else
    {
        sendFrame.addr = Sensor::fromAddrMap[SN];
        sendFrame.addr_len = sizeof(sendFrame.addr);
        sendFrame.dataLength = 2;
        sendFrame.data = sendData;
        Sensor::UDP.Send(&sendFrame);
        std::cout << "Calibrate signal  send to " << SN << "." << std::endl;
    }
}

void Sensor::quitSensor(std::string SN)
{
    SockUDPFrame_t sendFrame;
    char sendData[2] = {'$', 'Q'};
    if (Sensor::fromAddrMap.find(SN) == Sensor::fromAddrMap.end())
    {
        std::cout << "Quit failed! (sensor " << SN << " is not connected)" << std::endl;
    }
    else
    {
        sendFrame.addr = Sensor::fromAddrMap[SN];
        sendFrame.addr_len = sizeof(sendFrame.addr);
        sendFrame.dataLength = 2;
        sendFrame.data = sendData;
        Sensor::UDP.Send(&sendFrame);
        std::cout << "Quit signal send to " << SN << "." << std::endl;
    }
}

}
