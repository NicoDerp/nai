#!/usr/bin/python3

import numba
import math
import time
import numpy as np

@numba.njit
def doSomething(a, c):
    a[0] += c.f(a[1])



class MyClass:
    def __init__(self):
        pass

    @staticmethod
    @numba.njit
    def f(x):
        return x**2


a = np.array([1, 2, 3])
print(f"a before: {a}")

# Compilation + execution
start = time.time()
doSomething(a, MyClass)
end = time.time()
print(f"Elapsed (with compilation) = {end-start:.10f}")
print(f"a after: {a}")

#start = time.time()
#doSomething(a)
#end = time.time()
#print(f"Elapsed (without compilation) = {end-start:.10f}")

