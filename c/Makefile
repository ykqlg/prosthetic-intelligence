CC = gcc
CFLAGS = -I/usr/lib/include -g
LDFLAGS = -L/usr/lib -lwiringPi
FILENAME = main
UI = ui
GTKLDFLAGS = `pkg-config --cflags --libs gtk+-3.0`

SRC = main.c BMI088driver.c BMI088Middleware.c
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

.PHONY: ui
ui:
	$(CC) ui.c -o ui $(GTKLDFLAGS)