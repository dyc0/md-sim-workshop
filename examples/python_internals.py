"""Part 4: why is pure Python slow? A look under the hood.

1. `dis` shows the bytecode instructions Python runs for a tiny function:
   every `+` and `*` is a separate instruction that checks types at run time.
2. `sys.getsizeof` shows that a Python number is a whole object (with a type
   and a reference count), while NumPy stores bare 8-byte numbers side by side.

This script only prints; it makes no figures.
"""

# ---- Parameters (edit these) ----
N_NUMBERS = 1000  # length of the list and of the array we compare
# ---- End of parameters ----

import dis
import sys

import numpy as np


def f(a, b, c):
    return a + b * c


print("Bytecode of  def f(a, b, c): return a + b * c")
print("-" * 50)
dis.dis(f)

print()
print("Memory used by single Python objects (bytes):")
print(f"  a float, 1.0:  {sys.getsizeof(1.0)}")
print(f"  an int, 1:     {sys.getsizeof(1)}")

numbers = [float(i) for i in range(N_NUMBERS)]
list_size = sys.getsizeof(numbers) + sum(sys.getsizeof(x) for x in numbers)
array = np.arange(N_NUMBERS, dtype=np.float64)

print()
print(f"{N_NUMBERS} floats:")
print(f"  Python list (the list itself plus every float object): {list_size} bytes")
print(f"  NumPy array, nbytes:                                   {array.nbytes} bytes")
print(f"  NumPy itemsize (bytes per number):                     {array.itemsize}")
print(f"  The list uses {list_size / array.nbytes:.1f} times more memory.")
