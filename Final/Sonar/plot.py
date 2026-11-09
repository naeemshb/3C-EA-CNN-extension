import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)

def relu(z):
    return np.maximum(0, z)

# Ideal missingness effect: suppress output when missing
y_m0 = relu(x)                 # observed
y_m1 = 0.25 * relu(x)          # missing -> much smaller activation (down-weight imputed)

plt.figure(figsize=(7.5,4.5))
plt.plot(x, y_m0, label="m=0 (observed)")
plt.plot(x, y_m1, label="m=1 (missing/imputed)")
plt.title("Ideal Effect of Missing Flag (confidence fixed, e.g., c=1)")
plt.xlabel("x (value)")
plt.ylabel("activation(x)")
plt.legend()
plt.grid(True)
plt.show()

# Ideal confidence effect: output scales smoothly with confidence
c_high, c_low = 1.0, 0.3
y_c_high = c_high * relu(x)
y_c_low  = c_low  * relu(x)

plt.figure(figsize=(7.5,4.5))
plt.plot(x, y_c_high, label="c=1.0 (high confidence)")
plt.plot(x, y_c_low, label="c=0.3 (low confidence)")
plt.title("Ideal Effect of Confidence (missing fixed, e.g., m=0)")
plt.xlabel("x (value)")
plt.ylabel("activation(x)")
plt.legend()
plt.grid(True)
plt.show()
