import matplotlib.pyplot as plt
from collections import deque
import subprocess

# 创建一个双端队列用于存储传感器数据
data = deque(maxlen=10)

# 启动C语言项目并通过管道获取其输出
proc = subprocess.Popen('./main', stdout=subprocess.PIPE)

# 创建一个空的图表
fig, ax = plt.subplots()
line, = ax.plot([])

# 定义一个函数用于更新图表
def update_plot():
    global data
    sensor_data = int(proc.stdout.readline().decode().strip())
    data.append(sensor_data)
    line.set_ydata(data)
    return line,

# 设置图表属性
ax.set_ylim(0, 100)
ax.set_xlim(0, 9)
ax.grid()

# 启动动画更新图表
ani = animation.FuncAnimation(fig, update_plot, blit=True, interval=1000)

# 显示图表
plt.show()
