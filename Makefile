CC=g++
CFLAGS = -O0 -g -Wall # -I./ -I$(FFTW_INC) -I/u/shared/programs/x86_64/openmpi/1.8.3/gnu/4.9.2/torque/include/ -std=c99
LIBS = # -L -lm		
EXE = TWO-NN.x
SRC = TWO-NN_userfriendly.cc 
HEAD = 
OBJ = $(SRC:.c=.o)


all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $^ $(LIBS) -o $@

%.o : %.cc $(HEAD)
	$(CC) $(CFLAGS) -c $<

flush:
	rm -f *.dat *.btr

clean: 
	rm -f *.o *.x *~ *.dat *.gpl

run:
	./TWO-NN.x -input datasets/cauchy20 -coord


.PHONY: clean flush run