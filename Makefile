CC = gcc
CFLAGS = -I/usr/lib/include -g
LDFLAGS = -L/usr/lib -lwiringPi

SRC = main.c BMI088driver.c
TARGET = main

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: gdb
gdb: $(TARGET)
	gdb ./$(TARGET)