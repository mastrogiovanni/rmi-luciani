#!/bin/bash

docker run --rm -w /app -v /data/MASTROGIOVANNI/:/data/MASTROGIOVANNI -v $(pwd):/app mastrogiovanni/rmi-python3 python test.py

