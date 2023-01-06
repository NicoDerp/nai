
import importlib



numba_installed = importlib.find_loader("numba") is not None
if numba_installed:
    import numba
    def nnjit(func):
        jitted = numba.njit(func, fastmath=True)
        def wrapper(*args, **kwargs):
            return jitted(*args, **kwargs)
        return wrapper
else:
    def nnjit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper



# Credit to TeaCoast
@nnjit
def random_exclusion(start, stop, excluded):
    """Function for getting a random number with some numbers excluded"""
    #excluded = set(excluded) # if input is set then not needed
    value = random.randint(start, stop - len(excluded)) # Or you could use randrange
    for exclusion in tuple(excluded):
        if value < exclusion:
            break
        value += 1
    return value




