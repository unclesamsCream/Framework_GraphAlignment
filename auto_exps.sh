#!/bin/bash

# python workexp.py with cons_small load=[568] verbose=True
# python workexp.py with tuning13 verbose=True
# python workexp.py with tuning14 load=[643] verbose=True

taskset --cpu-list 1-14 python workexp.py with synth_inflexion verbose=True
taskset --cpu-list 1-14 python workexp.py with synth_ncomp verbose=True
taskset --cpu-list 1-14 python workexp.py with synth_benchmarks verbose=True
taskset --cpu-list 1-14 python workexp.py with synth_benchmarks_grampa verbose=True
taskset --cpu-list 1-14 python workexp.py with medium_size verbose=True


# python workexp.py with real verbose=True
# python workexp.py with real load=[661] verbose=True

# python workexp.py with rsc_small verbose=True
# python workexp.py with rsc_small2 load=[649] verbose=True

# for i in {1..5}
# do
# 	python workexp.py with cornell_n0 verbose=True
# done
# for i in {1..5}
# do
# 	python workexp.py with cornell_n05 verbose=True
# done
