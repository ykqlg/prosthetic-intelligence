CC = gcc
SRC = main_index.c 
TARGET = main_index
GTKLDFLAGS = `pkg-config --cflags --libs gtk+-3.0`

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) 

.PHONY: clean
clean:
	rm -f $(TARGET)

.PHONY: run
run: $(TARGET)
	./$(TARGET)


.PHONY: ui
ui:
	$(CC) ui.c -o ui $(GTKLDFLAGS)

.PHONY: gdb
gdb: $(TARGET)
	gdb ./$(TARGET)

# so: 
	$(CC) -std=c11 -Wall -Wextra -pedantic -shared -fPIC main.c -o main.so