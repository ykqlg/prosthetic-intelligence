CC = gcc
SRC = main.c 
TARGET = main

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) 

.PHONY: clean
clean:
	rm -f $(TARGET)

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: gdb
gdb: $(TARGET)
	gdb ./$(TARGET)

.PHONY: so
so: 
	$(CC) -std=c11 -Wall -Wextra -pedantic -shared -fPIC main.c -o main.so