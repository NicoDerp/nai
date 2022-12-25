#!/usr/bin/python3

import numba
import math
import time

@numba.njit(fastmath=True)
def doSomething(a, b):
    s = a/b
    for i in range(a):
        s *= math.e ** math.sqrt(i / a * b)
    return s ** s

def normalDoSomething(a, b):
    s = 0
    for i in range(a):
        s += math.e ** math.sqrt(i / a * b)
    return s

@numba.experimental.jitclass([
    ("a", numba.int32),
    ("b", numba.int32)
])
class MyClass(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def doSomething(self):
        return doSomething(self.a, self.b)

# Compilation + execution
start = time.time()
doSomething(50, 100)
end = time.time()
print(f"Elapsed (with compilation) = {end-start:.10f}")

start = time.time()
doSomething(50, 100)
end = time.time()
print(f"Elapsed (without compilation) = {end-start:.10f}")

start = time.time()
normalDoSomething(50, 100)
end = time.time()
print(f"Elapsed (no numba) = {end-start:.10f}")
