import time
import scipy.signal as signal
import spidev
import struct

class SpiManager:
    def __init__(self):
        self.spi = spidev.SpiDev()
        self.spi.open(0, 1)
        self.spi.max_speed_hz = 500000
        self.accel_init()

    def get_spi(self):
        return self.spi
    
    def close_spi(self):
        self.spi.close()
        return  # Close the SPI connection
    
    def read_multi_reg(self, reg, length):
        buf = []
        for i in range(length):
            send_data = [reg | 0x80, 0]
            recv_data = self.spi.xfer2(send_data)
            buf.append(recv_data[1])
            reg += 1
        return buf

    def write_reg(self, reg, data):
        send_data = [reg, data]
        self.spi.writebytes(send_data)

    def accel_z(self):
        data = self.read_multi_reg(0x16, 3)
        accel_sen = 1000*2*1.5/32768
    
        z = (struct.unpack('<h', bytes(data[1:3]))[0])*accel_sen
        
        return z
    
    def accel_read(self):
        data = self.read_multi_reg(0x12, 7)
        accel_sen = 1000*2*1.5/32768
        
        # 这种写法会导致溢出
        # x = (data[2]<<8 | data[1])*accel_sen
        # y = (data[4]<<8 | data[3])*accel_sen
        # z = (data[6]<<8 | data[5])*accel_sen
        
        x = (struct.unpack('<h', bytes(data[1:3]))[0])*accel_sen
        y = (struct.unpack('<h', bytes(data[3:5]))[0])*accel_sen
        z = (struct.unpack('<h', bytes(data[5:7]))[0])*accel_sen
        
        return [x,y,z]
    
    def accel_init(self):
        # self.soft_reset(spi)
        write_BMI088_accel_reg_data_error = [
        [0x7D, 0x04, 'BMI088_ACC_PWR_CTRL_ERROR'],
        [0x7C, 0x00, 'BMI088_ACC_PWR_CONF_ERROR'],
        [0x40, 0xAC, 'BMI088_ACC_CONF_ERROR'],
        [0x41, 0x00, 'BMI088_ACC_RANGE_ERROR'],
        [0x53, 0x08, 'BMI088_INT1_IO_CTRL_ERROR'],
        [0x58, 0x04, 'BMI088_INT_MAP_DATA_ERROR']
    ]
    
        res = self.read_multi_reg(0x00,1)
        time.sleep(0.00015)
        res = self.read_multi_reg(0x00,1)
        time.sleep(0.00015)
        
        for reg_data in write_BMI088_accel_reg_data_error:
            self.write_reg(reg_data[0],reg_data[1])
            time.sleep(0.00015)
            
    def soft_reset(self):
        self.write_reg(0x7E,0xB6)
        time.sleep(1)
        

    def self_test(self):
        spi = self.spi
        accel_pos = [0.0, 0.0, 0.0]
        accel_neg = [0.0, 0.0, 0.0]

        self.write_reg(spi,0x41, 0x3)
        self.write_reg(spi,0x40, 0xA7)
        time.sleep(0.003)
        self.write_reg(spi,0x6D, 0x0D)
        time.sleep(0.05)
        
        accel_pos = self.accel_read()
        self.write_reg(spi,0x6D, 0x09)
        time.sleep(0.05)
        
        accel_neg = self.accel_read()
        self.write_reg(0x6D, 0x00)

        result = [0.0, 0.0, 0.0]
        for i in range(3):
            result[i] = accel_pos[i] - accel_neg[i]

        if not (result[0] >= 1000 and result[1] >= 1000 and result[2] >= 500):
            print(f"{result[0],result[1],result[2] }")
            print("Self Test: Not satisfy the expected values!!!")
            exit(1)
        time.sleep(0.05)
        self.soft_reset(spi)



def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff,fs=fs, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y
