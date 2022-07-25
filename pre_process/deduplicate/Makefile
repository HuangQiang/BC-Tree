SRCS=util.cc de_duplicate.cc main.cc
OBJS=$(SRCS:.cc=.o)

CXX=g++ -std=c++11
CPPFLAGS=-w -O3

.PHONY: clean

all: $(OBJS)
	$(CXX) ${CPPFLAGS} -o dedup $(OBJS)

util.o: util.h

de_duplicate.o: de_duplicate.h

main.o:

clean:
	-rm $(OBJS) dedup
