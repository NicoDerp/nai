#!/usr/bin/python3

import numba
import math
import numpy as np
import time

from multiprocessing import Pool


@numba.njit(fastmath=True)
def doSomething(n):
    return np.sqrt(n) ** np.sqrt(2*n+1)


@numba.njit(parallel=True, fastmath=True)
def numbaParallel(nJobs):
    s = 0
    for i in numba.prange(nJobs):
        s += doSomething(i)
    return s

def manualParallel(nJobs):
    inputs = []
    for i in range(nJobs):
        inputs.append(i)

    pool = Pool(nJobs)

    results = pool.map(doSomething, inputs)
    s = sum(results)
    return s

def noParallel(nJobs):
    s = 0
    for i in range(nJobs):
        s += doSomething(i)
    return s



@numba.njit
def test(layers):
    a = np.array([1, 2, 3])
    layers[0] = a

l = np.array([[0, 1, 2], [3, 4, 5]])
print(l)

test(l)

print(l)

exit()

nJobs = 100

# JIT compile functions first
doSomething(0)
numbaParallel(0)


start = time.time()
print(manualParallel(nJobs))
end = time.time()
print(f"Took {end-start:.10f}s for manual parallel.")


start = time.time()
print(noParallel(nJobs))
end = time.time()
print(f"Took {end-start:.10f}s for no parallel.")


start = time.time()
print(numbaParallel(nJobs))
end = time.time()
print(f"Took {end-start:.10f}s for numba parallel.")


