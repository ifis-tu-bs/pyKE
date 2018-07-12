#!/usr/bin/env bash

g++ -std=c++11 -Wall $1 -fPIC -shared -o $2 -pthread -O3 -march=native
