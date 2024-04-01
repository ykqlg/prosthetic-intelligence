from driver import *
import time
def main():
    count = 1000
    st = time.time()
    for i in range(count):
        z= spi.accel_z()
    et = time.time()

    print(f"average time: {(et-st)*1000/count} (ms) ")
    # average time: 0.18546605110168457 (ms)


if __name__ == '__main__':
    spi = SpiManager()
    main()
    spi.close_spi()