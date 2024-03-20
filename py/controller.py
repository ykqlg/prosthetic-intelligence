import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.LOW)
blinks = 0
print('开始闪烁')
while (blinks < 5):
    GPIO.output(18, GPIO.HIGH)
    time.sleep(1.0)
    GPIO.output(18, GPIO.LOW)
    time.sleep(1.0)
    blinks = blinks + 1
GPIO.output(18, GPIO.LOW)
GPIO.cleanup()
print('结束闪烁')


#################
import spidev

def BMI088_read_multi_reg(spi, reg, len):
    buf = []
    for i in range(len):
        # 构建发送的数据
        send_data = [reg | 0x80, 0]
        # 通过SPI总线发送数据并接收回复
        recv_data = spi.xfer(send_data)
        # 将接收到的数据添加到缓冲区中
        buf.append(recv_data[1])
        # 更新寄存器地址
        reg += 1
    return buf

# 初始化SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # 指定SPI设备，0表示SPI总线，0表示片选信号线

# 读取多个寄存器的值
reg = 0x00  # 起始寄存器地址
len = 5  # 要读取的寄存器数量
data = BMI088_read_multi_reg(spi, reg, len)
print("Multi reg:", [hex(d) for d in data])

# 关闭SPI
spi.close()