OBJS = heatmap.o bmp.o interpolation.o colors.o
TARGET = heatmap

IDIR = .

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -I$(IDIR)
LDFLAGS = -lm


.PHONY: clean all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean:
	rm $(TARGET) $(OBJS) *~
