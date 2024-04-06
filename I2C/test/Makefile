CC = gcc
SRC = mylib.c 
TARGET = mylib

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC)

.PHONY: so
so: 
	$(CC) -std=c11 -Wall -Wextra -pedantic -shared -fPIC $(SRC) -o $(TARGET).so

.PHONY: clean
clean:
	rm -f $(TARGET)

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: gdb
gdb: $(TARGET)
	gdb ./$(TARGET)

