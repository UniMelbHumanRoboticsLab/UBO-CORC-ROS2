# FLNL client-server (see https://github.com/vcrocher/libFLNL)
#
# Vincent Crocher - Unimelb - 2022, 2024
#
# Apache 2.0 License

import socket
import sys
import threading
import struct
import time


def Checksum(b):
    ck = 0
    for i in range(2, len(b)-1):
        ck = ck ^ b[i]
    return ck



#TODO: extend with cmd associate values reception

class FLNL:
    MAXVALS=31

    def __init__(self):
        self.newCmdRcv = False
        self.newValsRcv = False
        self.CmdRcv = ""
        self.Connected = False
        self.receiving=False


    def __del__(self):
        self.Close()

    def IsConnected(self):
        return self.Connected

    def SendValues(self, vals):
        if(self.Connected):
            if(len(vals)>self.MAXVALS):
                print('FLNL: Error, too many values to send')
                return

            #Build packet header to send:
            tosend=bytearray(255);
            tosend[0]=ord('V');
            tosend[1]=len(vals);

            #Pack double values
            i=2;
            for val in vals:
                val_b=bytearray(struct.pack("d", val))
                for byte in val_b:
                    tosend[i]=byte
                    i=i+1

            tosend[255-1]=Checksum(tosend);

            #send
            try:
                self.connection.sendall(tosend)
            except (BrokenPipeError, ConnectionResetError):
                self.Connected=False
                self.receiving=False
                return False

            return True



    def SendCmd(self, cmd, vals=None):
        if(self.Connected):
            if vals is None:
                vals = []

            #Build packet header to send:
            tosend=bytearray(255);
            tosend[0]=ord('C');
            tosend[1]=len(vals);

            #Command
            cmd_bytes=cmd.encode()
            i=2
            for byte in cmd_bytes:
                tosend[i]=byte
                i=i+1
            for k in range(4-len(cmd_bytes)):
                tosend[i]=0
                i=i+1

            #Pack double values
            for val in vals:
                val_b=bytearray(struct.pack("d", val))
                for byte in val_b:
                    tosend[i]=byte
                    i=i+1

            tosend[255-1]=Checksum(tosend);

            #send
            try:
                self.connection.sendall(tosend)
            except (BrokenPipeError, ConnectionResetError):
                self.Connected=False
                self.receiving=False
                return False

            return True


    def ProcessRcvValues(self, data, nbvals):
        self.ValsRcv=[]
        for i in range(0, 8*nbvals, 8):
            try:
                self.ValsRcv.append(struct.unpack('d', bytearray(data[i:i+8]))[0])
            except Exception as e:
                #If for some reason one value is corrupted, give up all sequence
                nbvals = 0
                self.ValsRcv=[]
                return

    def recFct(self):
        while self.receiving:
            try:
                data = self.connection.recv(255)
            except BlockingIOError:
                continue
            except (BrokenPipeError, ConnectionResetError):
                self.Connected=False
                self.receiving=False
                return False

            if data:
                if data[0]==ord('C'):
                    #print('Cmd: '+str(data[2:5]))
                    self.newCmdRcv = True
                    self.CmdRcv = data[2:5].decode("utf-8")
                    nbvals = data[1]
                    self.ProcessRcvValues(data[6:6+8*nbvals], nbvals)
                if data[0]==ord('V'):
                    nbvals = data[1]
                    #print('Values ('+str(nbvals)+')')
                    self.ProcessRcvValues(data[2:2+8*nbvals], nbvals)
                    if nbvals>0:
                        self.newValsRcv = True


    def IsValues(self):
        return self.newValsRcv

    def GetValues(self):
        if(self.newValsRcv):
            self.newValsRcv = False
            return self.ValsRcv
        else:
            return []

    def IsAnyCmd(self):
        return self.newCmdRcv

    def IsCmd(self, cmd):
        if self.newCmdRcv and self.CmdRcv==cmd:
            self.newCmdRcv = False;
            return True
        else:
            return False

    def IsCmdHas(self, cmd):
        if self.newCmdRcv and cmd in self.CmdRcv:
            self.newCmdRcv = False;
            return True
        else:
            return False

    def GetCmd(self):
        if(self.newCmdRcv):
            self.newCmdRcv = False;
            return self.CmdRcv
        else:
            return ""


class FLNLServer(FLNL):

    def WaitForClient(self, ip="127.0.0.1", port=2048):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip, port)
        self.sock.bind(server_address)
        self.sock.listen(1)

        # Wait for a connection (blocking)
        print('FLNLServer: Waiting for a connection ('+ip+':'+str(port)+')...')
        self.connection, client_address = self.sock.accept()
        print('FLNLServer: Client connected!');
        self.Connected = True

        # Create reception thread
        self.receiving = True
        recPr = threading.Thread(target=self.recFct, daemon=True)
        recPr.start()


    def Close(self):
        self.receiving=False
        time.sleep(0.5)
        if(self.Connected):
            self.connection.close()
        self.Connected=False


class FLNLClient(FLNL):

    def Connect(self, ip="127.0.0.1", port=2048):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(5.)
        server_address = (ip, port)

        # Connection
        print('FLNLClient: connecting to ('+ip+':'+str(port)+')...')
        try:
            self.sock.connect(server_address)
        except Exception as e:
            print('FLNLClient: Connection failed! (', e, ')');
            self.Connected = False
            self.sock.close()
            return self.Connected

        print('FLNLClient: Client connected!');
        self.connection = self.sock
        self.Connected = True

        # Create reception thread
        self.receiving=True
        recPr = threading.Thread(target=self.recFct, daemon=True)
        recPr.start()

        return self.Connected


    def Close(self):
        self.receiving=False
        time.sleep(0.5)
        if(self.Connected):
            self.connection.close()
        self.Connected=False
