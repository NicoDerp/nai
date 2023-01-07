
import numba


def nnjit(func):
    jitted = numba.njit(func, fastmath=True)
    return jitted


@nnjit
def myFunc(a, b):
    s = 0
    for n in a:
        s += n * b
    return s


a = (1, 2)
b = 1


print(myFunc(a, b))
print(myFunc(a, b))


