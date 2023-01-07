
import importlib
import random


numba_installed = importlib.find_loader("numba") is not None
if numba_installed:
    import numba
    def nnjit(func):
        jitted = numba.njit(func, fastmath=True)
        #def wrapper(*args, **kwargs):
        #    return jitted(*args, **kwargs)
        #return wrapper
        return jitted
else:
    def nnjit(func):
        #def wrapper(*args, **kwargs):
        #    return func(*args, **kwargs)
        #return wrapper
        return func



# Credit to TeaCoast
@nnjit
def random_exclusion(start, stop, excluded):
    """Function for getting a random number with some numbers excluded"""
    #excluded = set(excluded) # if input is set then not needed
    value = random.randint(start, stop - len(excluded)) # Or you could use randrange
    #for exclusion in tuple(excluded):
    for exclusion in excluded:
        if value < exclusion:
            break
        value += 1
    return value




