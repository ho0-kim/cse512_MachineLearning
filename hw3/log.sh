#!/bin/bash

for i in {1..10}
do
	echo "####    Runing   Question3 python code Trial $i"
	python3 hw3_q3.py | grep Test >> hw3_q3.log
done
