#!/bin/bash

for i in {1..10}
do
	echo "####    Runing   Question2 python code $i th trial "
	python3 hw3_q2.py >> hw3_q2.log
done
