#!/bin/bash

# python workexp.py with cons_small load=[568] verbose=True
python workexp.py with tuning13 verbose=True
python workexp.py with tuning14 load=[643] verbose=True

# for i in {1..5}
# do
# 	python workexp.py with cornell_n0 verbose=True
# done
# for i in {1..5}
# do
# 	python workexp.py with cornell_n05 verbose=True
# done
