
import numpy as np
import time

l = list(range(1000000))

start = time.time()

a = np.array(l)

end = time.time()
timespent = end - start
print(f"Took {timespent:.20f}s")


