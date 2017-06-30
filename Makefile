CPPFLAGS += -std=c11 -Ofast -Wall -mavx2 -I/usr/local/include
LDFLAGS += -L/usr/local/lib -lm4ri -lm
.PHONY: all clean

all: fast_muls 

fast_muls: fast_muls.o

fast_muls.o: fast_muls.c

clean:
	rm *.o
