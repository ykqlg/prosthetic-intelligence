""" Python wrapper for the C shared library mylib"""
import sys, platform
import ctypes, ctypes.util

# print("mylib_py")
mylib_path = "./mylib.so"
try:
    mylib = ctypes.CDLL(mylib_path)
except OSError:
    print("Unable to load the system C library")
    sys.exit()

BUFFER_SIZE = 6
# 定义结构体 SensorInfo
class SensorInfo(ctypes.Structure):
    _fields_ = [("sensorIndex", ctypes.c_int),
                ("i2cFile", ctypes.c_int),
                ("msgBuffer", ctypes.c_ubyte * BUFFER_SIZE),
                ("startTime", ctypes.c_char * 100)]

# 定义指针类型 pSensor
pSensor = ctypes.POINTER(SensorInfo)


# 声明 C 函数 writeRegister
writeRegister = mylib.writeRegister
writeRegister.argtypes = [ctypes.c_int, ctypes.c_ubyte, ctypes.c_ubyte]
writeRegister.restype = None

# 声明 C 函数 readRegOneByte
readRegOneByte = mylib.readRegOneByte
readRegOneByte.argtypes = [ctypes.c_int, ctypes.c_ubyte]
readRegOneByte.restype = ctypes.c_ubyte

# 声明 C 函数 initializeByteStreaming
initializeByteStreaming = mylib.initializeByteStreaming
initializeByteStreaming.argtypes = [pSensor, ctypes.c_ubyte, ctypes.c_int]
initializeByteStreaming.restype = None

# 声明 C 函数 setup
setup = mylib.setup
# setup.argtypes = [ctypes.POINTER(pSensor)]
setup.argtypes = [pSensor]
setup.restype = None

# 声明 C 函数 loop
loop = mylib.loop
loop.argtypes = [pSensor]
loop.restype = None

# 声明 C 函数 sensorThread
sensorThread = mylib.sensorThread
sensorThread.argtypes = [ctypes.c_void_p]
sensorThread.restype = ctypes.c_void_p

# 声明 C 函数 Min
Min = mylib.Min
Min.argtypes = [ctypes.c_int, ctypes.c_int]
Min.restype = ctypes.c_int

# 声明 C 函数 get_positive_number
get_positive_number = mylib.get_positive_number
get_positive_number.argtypes = []
get_positive_number.restype = ctypes.c_int

# 声明 C 函数 initSensors
initSensors = mylib.initSensors
initSensors.argtypes = [pSensor]
initSensors.restype = None

read_dataZ = mylib.read_dataZ
read_dataZ.argtypes = [pSensor]
read_dataZ.restype = ctypes.c_int16


