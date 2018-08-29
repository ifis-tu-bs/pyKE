#!/usr/bin/env bash

g++ $1 -fPIC -shared -o $2 -pthread -O3 -march=native
