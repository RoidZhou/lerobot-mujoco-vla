#! /usr/bin/python3
import ipaddress
import os
import socket
import struct
import threading
import time


class UDP_MC_Manager:
    def __init__(self, callback=None, isSender=False, ip="", group="224.0.0.1", port=8083, frequency=1000):
        self.callback = callback

        self.isSender = isSender
        self.interval = 1.0 / frequency

        self.af_inet = None
        if group is not None and group != "" and ipaddress.ip_address(group) in ipaddress.ip_network("224.0.0.0/4"):
            self.group = group
        elif self.isSender:
            print("[UDP Manager] Invalid multicast group address, should be in 224.0.0.0/4")
            return
        else:
            self.group = ""

        self.ip = ip
        self.port = port
        self.addr = (self.group, self.port)
        self.running = False

    def start(self):
        self.af_inet = socket.AF_INET  # ipv4
        self.sockUDP = socket.socket(self.af_inet, socket.SOCK_DGRAM)

        if self.isSender:
            self.roleName = "Sender"
            self.sockUDP.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            self.sockUDP.bind((self.ip, 0))
        else:
            self.roleName = "Receiver"
            self.sockUDP.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sockUDP.bind(("", self.port))
            self.sockUDP.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 212992)
            mreq = struct.pack("4s4s", socket.inet_aton(self.group), socket.inet_aton(self.get_interface_ip(self.ip)))
            self.sockUDP.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        print("[UDP Manager]", self.roleName, "at:", self.group, ":", self.port)

        self.running = True
        if not self.isSender and self.callback is not None:
            self.thread = threading.Thread(target=self.receive, args=())
            self.thread.daemon = True
            self.thread.start()  # 打开收数据的线程

    def receive(self):
        while self.running:
            time.sleep(self.interval)
            while self.running:
                try:
                    recvData, recvAddr = self.sockUDP.recvfrom(65535)  # 等待接受数据
                except:
                    break
                if not recvData:
                    break
                self.callback(recvData, recvAddr)

    def send(self, data):
        if self.isSender:
            self.sockUDP.sendto(data, self.addr)

    def close(self):
        self.running = False

    def get_interface_ip(self, ip):
        s = socket.socket(self.af_inet, socket.SOCK_DGRAM)
        s.connect((ip, 0))
        addr = s.getsockname()
        s.close()
        return addr[0]


class TCP_Client_Manager:
    def __init__(self, callback=None, ip="", port=8083, frequency=1000):
        self.callback = callback
        self.interval = 1.0 / frequency

        self.af_inet = None
        self.ip = ip
        self.port = port
        self.addr = (self.ip, self.port)
        self.running = False

    def start(self):
        self.af_inet = socket.AF_INET  # ipv4
        self.sockTCP = socket.socket(self.af_inet, socket.SOCK_STREAM)

        self.roleName = "Client"
        self.sockTCP.connect(self.addr)

        print("[TCP Manager]", self.roleName, "at:", self.ip, ":", self.port)

        self.running = True
        if self.callback is not None:
            self.thread = threading.Thread(target=self.receive, args=())
            self.thread.daemon = True
            self.thread.start()  # 打开收数据的线程

    def receive(self):
        while self.running:
            time.sleep(self.interval)
            while self.running:
                try:
                    recvData, recvAddr = self.sockTCP.recvfrom(65535)  # 等待接受数据
                except:
                    break
                if not recvData:
                    break
                self.callback(recvData, recvAddr)

    def send(self, data):
        self.sockTCP.sendto(data, self.addr)

    def close(self):
        self.sockTCP.close()
        self.running = False


def crc16(data: bytes, size: int):
    crc = 0xFFFF
    for i in range(0, size):
        crc ^= data[i]
        for j in range(0, 8):
            if (crc & 0x0001) > 0:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc = crc >> 1
    return crc & 0xFFFF


class Tac3d_Manager:
    CMD_NONE = 0
    CMD_GET_ID = 1
    CMD_GET_RUN_IP = 2
    CMD_GET_CONF_IP = 3
    CMD_SET_CONF_IP = 4
    CMD_GET_TIME = 5
    CMD_SET_TIME = 6
    CMD_IF_RESTART = 11
    CMD_SYS_SHUTDOWN = 12
    CMD_SYS_RESTART = 13
    CMD_GET_CONF = 21
    CMD_ADD_CONF = 22
    CMD_DEL_CONF = 23
    CMD_RUN_TAC3D = 24
    CMD_STAT_TAC3D = 25
    CMD_STOP_TAC3D = 26
    CMD_GET_LOG = 27
    CMD_CLEAR_LOG = 28
    CMD_PING = 255

    TIMEOUT = 1.0

    def __init__(self, ip):
        """
        ip: ip of interface
        """
        self.r_ctx = UDP_MC_Manager(self._recv_cb, False, ip, "239.0.2.102", 60033)
        self.r_ctx.start()
        self.t_ctx = UDP_MC_Manager(None, True, ip, "239.0.2.101", 60032)
        self.t_ctx.start()
        self.connected = False
        self.pack_list = []
        self.sn_list = []

    def connect_server(self, ip):
        """
        ip: get from get_run_ip
        """
        self.c_ctx = TCP_Client_Manager(self._recv_cb, ip, 60030)
        self.c_ctx.start()
        self.connected = True

    def disconnect_server(self):
        if self.connected:
            self.c_ctx.close()
        self.connected = False

    def _recv_cb(self, recv_data: bytes, recv_addr):
        if len(recv_data) < 8:
            return
        # split
        head = recv_data[0:8]
        data = recv_data[8:]
        # unpack header
        cmd, sub_cnt, size, _crc16 = struct.unpack("<HHHH", head)
        # check size
        if size + 8 != len(recv_data):
            print("size is not equal: %d vs %d" % (size + 8, len(recv_data)))
            print(data)
            return
        # check crc16
        __crc16 = crc16(data, size)
        if _crc16 != __crc16:
            print("crc16 is not equal: %d vs %d" % (__crc16, _crc16))
            return
        # fill output
        self.pack_list.append((cmd, sub_cnt, data))

    def get_tp_id(self):
        # make pack
        data = struct.pack("<")
        cmd = self.CMD_GET_ID
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # receive
        sub_cnt = 0
        num = 0
        tp_list = []
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    tp = pdata[0:64].split(b'\x00', 1)[0].decode()
                    print(tp)
                    tp_list.append(tp)
                    num += 1
                    # remove unpacked data
                    self.pack_list.remove(pack)
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                return (num, tp_list)

    def get_config(self):
        # make pack
        data = struct.pack("<")
        cmd = self.CMD_GET_CONF
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        num = 0
        cfg_num = 0
        cfg_list = []
        sn_list = []
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    if sub_cnt == 0:
                        head = pdata[0:4]
                        (cfg_num,) = struct.unpack("<i", head)
                        print("number %d" % cfg_num)
                        pdata = pdata[4:]
                    s_num = int(len(pdata) / 128)
                    for i in range(0, s_num):
                        cfg = pdata[0:64].split(b'\x00', 1)[0].decode()
                        print(cfg)
                        cfg_list.append(cfg)
                        pdata = pdata[64:]
                        sn = pdata[0:64].split(b'\x00', 1)[0].decode()
                        print(sn)
                        sn_list.append(sn)
                        pdata = pdata[64:]
                        num += 1
                    sub_cnt += 1
                    # remove unpacked data
                    self.pack_list.remove(pack)
                # check num to stop
                if num >= cfg_num:
                    return (cfg_num, cfg_list, sn_list)
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return (-1, [], [])

    def add_config(self, fn: str):
        # open config
        file = open(fn, "rb")
        file_size = os.path.getsize(fn)
        fdata = file.read(1370)
        sub_cnt = 0
        print("file size %d" % file_size)
        while fdata:
            # make pack
            if sub_cnt == 0:
                data = struct.pack("<i", file_size) + fdata
            else:
                data = fdata
            cmd = self.CMD_ADD_CONF
            size = len(data)
            _crc16 = crc16(data, size)
            head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
            # send
            self.c_ctx.send(head + data)
            # next pack
            sub_cnt += 1
            fdata = file.read(1380)
        # receive
        tstart = time.time()
        while True:
            should_break = False
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt + 1:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    should_break = True
                    break
            if should_break:
                break
            # wait
            time.sleep(0.002)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                break
        file.close()

    def delete_config(self, cfg: str):
        # make pack
        data = struct.pack("<64s", cfg.encode())
        cmd = self.CMD_DEL_CONF
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return

    def get_run_ip(self, tp_id: str):
        # make pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_GET_RUN_IP
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    ip = pdata[0:16].split(b'\x00', 1)[0].decode()
                    netmask = pdata[16:32].split(b'\x00', 1)[0].decode()
                    gateway = pdata[32:48].split(b'\x00', 1)[0].decode()
                    print("IP: %s, netmask: %s, gateway: %s" % (ip, netmask, gateway))
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return (ip, netmask, gateway)
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return ("", "", "")

    def get_config_ip(self, tp_id: str):
        # make pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_GET_CONF_IP
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    ip = pdata[0:16].split(b'\x00', 1)[0].decode()
                    netmask = pdata[16:32].split(b'\x00', 1)[0].decode()
                    gateway = pdata[32:48].split(b'\x00', 1)[0].decode()
                    print("IP: %s, netmask: %s, gateway: %s" % (ip, netmask, gateway))
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return (ip, netmask, gateway)
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return ("", "", "")

    def set_config_ip(self, tp_id: str, ip: str, netmask: str, gateway: str):
        # make pack
        data = struct.pack("<64s16s16s16s", tp_id.encode(), ip.encode(), netmask.encode(), gateway.encode())
        cmd = self.CMD_SET_CONF_IP
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return

    def interface_restart(self, tp_id: str):
        # make pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_IF_RESTART
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # this command will not send back data, we need ping to check connection
        time.sleep(0.1)
        # make ping pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_PING
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # ping
        self.t_ctx.send(head + data)
        # receive
        sub_cnt = 0
        cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            cnt += 1
            if cnt % 100 == 0:
                # ping again
                self.t_ctx.send(head + data)
            if tstart + self.TIMEOUT * 20 <= time.time():
                print("timeout")
                return

    def system_shutdown(self, tp_id: str):
        """
        user should stop the program after this command as the server is shutdown
        """
        # make pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_SYS_SHUTDOWN
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # this command will not send back data, just return
        time.sleep(1.0)

    def system_restart(self, tp_id: str):
        """
        user should re-init the client after this command as the connection will be down for a while
        """
        # make pack
        data = struct.pack("<64s", tp_id.encode())
        cmd = self.CMD_SYS_RESTART
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.t_ctx.send(head + data)
        # this command will not send back data, just return
        time.sleep(1.0)

    def get_time(self):
        # make pack
        data = struct.pack("<")
        cmd = self.CMD_GET_TIME
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    (local_time,) = struct.unpack("<d", pdata[0:8])
                    print("time: %f" % local_time)
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return local_time
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return 0.0

    def set_time(self, local_time: float):
        # make pack
        data = struct.pack("<d", local_time)
        cmd = self.CMD_SET_TIME
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return

    def run_tac3d(self, cfg: str, ip: str, port: int):
        # make pack
        data = struct.pack("<H64s64s", port, cfg.encode(), ip.encode())
        cmd = self.CMD_RUN_TAC3D
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    (video_id,) = struct.unpack("<i", pdata[0:4])
                    print("video_id %d" % video_id)
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return video_id
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT * 10 <= time.time():
                print("timeout")
                return -1

    def stat_tac3d(self, cfg: str):
        # make pack
        data = struct.pack("<64s", cfg.encode())
        cmd = self.CMD_STAT_TAC3D
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    (stat,) = struct.unpack("<i", pdata[0:4])
                    print("state %d" % stat)
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return bool(stat)
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return -1

    def stop_tac3d(self, cfg: str):
        # make pack
        data = struct.pack("<64s", cfg.encode())
        cmd = self.CMD_STOP_TAC3D
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT * 2 <= time.time():
                print("timeout")
                return

    def get_log(self, SN: str, fn: str):
        # open config
        file = open(fn, "wb")
        file_size = 0
        # make pack
        data = struct.pack("<64s", SN.encode())
        cmd = self.CMD_GET_LOG
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        file_pos = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # unpack
                    if sub_cnt == 0:
                        head = pdata[0:4]
                        (file_size,) = struct.unpack("<i", head)
                        print("file size %d" % file_size)
                        pdata = pdata[4:]
                    file.write(pdata)
                    file_pos += len(pdata)
                    sub_cnt += 1
                    # remove unpacked data
                    self.pack_list.remove(pack)
                # check num to stop
                if file_pos >= file_size:
                    file.close()
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                file.close()
                return

    def clear_log(self):
        # make pack
        data = struct.pack("<")
        cmd = self.CMD_CLEAR_LOG
        sub_cnt = 0
        size = len(data)
        _crc16 = crc16(data, size)
        head = struct.pack("<HHHH", cmd, sub_cnt, size, _crc16)
        # send
        self.c_ctx.send(head + data)
        # receive
        sub_cnt = 0
        tstart = time.time()
        while True:
            # search all packs
            for pack in self.pack_list:
                # check cmd and sub_count
                pcmd, pcnt, pdata = pack
                if cmd == pcmd and sub_cnt == pcnt:
                    # remove unpacked data and stop
                    self.pack_list.remove(pack)
                    return
            # wait
            time.sleep(0.01)
            if tstart + self.TIMEOUT <= time.time():
                print("timeout")
                return


if __name__ == "__main__":
    tac3d_manager = Tac3d_Manager("192.168.2.1")
    tp_num, tp_list = tac3d_manager.get_tp_id()
    tp_id = tp_list[0]
    # IP config
    ip, netmask, gateway = tac3d_manager.get_run_ip(tp_id)
    tac3d_manager.connect_server(ip)
    # Tac3D config
    tac3d_manager.add_config("test_conf.tcfg")
    cfg_num, cfg_list, sn_list = tac3d_manager.get_config()
    # tac3d
    tac3d_manager.run_tac3d(cfg_list[0], "192.168.2.1", 9988)
    tac3d_manager.stat_tac3d(cfg_list[0])
    time.sleep(30)
    tac3d_manager.stop_tac3d(cfg_list[0])
    tac3d_manager.get_log(sn_list[0], "log.zip")
    tac3d_manager.disconnect_server()
