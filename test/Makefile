# test suites for libopenke

all: query.0

query.0: query.0.c++
	g++ -std=c++14 -Wall -I../base -o $@ $^ -L.. -ldl

.PHONY: all
