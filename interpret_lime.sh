#!/bin/sh
cd interpret
python3 interpret_lime_linear.py 8 ./../
python3 interpret_lime_linear.py 16 ./../
cd ..