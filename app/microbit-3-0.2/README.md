# Microbit 3

A Microbit library for Python 3.

---

# Code Sample

```python
import microbit from Microbit

microbit = Microbit.Microbit("COM3") # The port that the microbit is connected to.

while True:
    microbit.readData()

    if microbit.pressed_a():
        print("A has been pressed")
```