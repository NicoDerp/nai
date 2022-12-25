#!/usr/bin/python3

import numba
import math
import time

@numba.njit(fastmath=True)
def doSomething(a, b):
    s = a/b
    a = Test(5)
    for i in range(a):
        s *= math.e ** a.other(math.sqrt(i / a * b))
    return s ** s

class Test:
    def __init__(self, n):
        self.n = n
    def other(self):
        return otherThing(self.n)
    @staticmethod
    @numba.njit(fastmath=True)
    def otherThing(n):
        return n**2 + 2*n - 1

# Compilation + execution
start = time.time()
doSomething(50, 100)
end = time.time()
print(f"Elapsed (with compilation) = {end-start:.10f}")

start = time.time()
doSomething(50, 100)
end = time.time()
print(f"Elapsed (without compilation) = {end-start:.10f}")

