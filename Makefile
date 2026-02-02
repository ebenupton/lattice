CC = gcc
CFLAGS = -O2 -Wall

all: lattice_encode lattice_decode test_bitstream

lattice_encode: lattice_encode.c lattice_common.h lattice_bitstream.h lattice_rans.h
	$(CC) $(CFLAGS) -o $@ lattice_encode.c -lm

lattice_decode: lattice_decode.c lattice_common.h lattice_bitstream.h lattice_rans.h
	$(CC) $(CFLAGS) -o $@ lattice_decode.c

test_bitstream: test_bitstream.c lattice_common.h lattice_bitstream.h lattice_rans.h
	$(CC) $(CFLAGS) -o $@ test_bitstream.c

clean:
	rm -f lattice_encode lattice_decode test_bitstream
	rm -f *.o

.PHONY: all clean
