CPPFLAGS += -std=c11 -Ofast -Wall -mavx2 -I/usr/local/include
LDFLAGS += -L/usr/local/lib -lm4ri -lm
.PHONY: all clean

all: fast_muls bro_main 

fast_muls: fast_muls.o

fast_muls.o: fast_muls.c

bro_main: bro_main.o bro_ska.o

bro_main.o: bro_main.c

bro_ska.o: bro_ska.c

clean:
	rm *.o
