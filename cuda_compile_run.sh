#!/bin/bash

cd ./build

make || exit 1
cd ./Linux-x86_64/
./popsift-demo --input-file ~/Downloads/sample_640×426.pgm
