# ------------------------------------------------------------------------------
#  Compile with C++ 17
# ------------------------------------------------------------------------------
SRCS=pri_queue.cc util.cc qalsh.cc rqalsh.cc main.cc
OBJS=${SRCS:.cc=.o}

CXX=g++-8 -std=c++17
# CXX=g++ -std=c++17
CPPFLAGS=-w -O3

.PHONY: clean

all: ${OBJS}
	${CXX} ${CPPFLAGS} -o p2h ${OBJS}

clean:
	-rm ${OBJS} p2h
