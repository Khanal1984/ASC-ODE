import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../legendre.txt")

#legendre_polynomials = data.iloc[:,1::2]

#Legendre polynomials
fig, ax = plt.subplots()

for col in data.columns[1::2]:
    plt.plot(data.iloc[:, 0], data[col], label=col)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-1,1)

ax.set_title("Legendre Polynommials")
ax.grid()
ax.legend()
plt.savefig("legendre_polynomials.png")

#Legendre polynomials derivatives
fig, ax = plt.subplots()

for col in data.columns[2::2]:
    plt.plot(data.iloc[:, 0], data[col], label=col)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-1,1)

ax.set_title("Legendre Polynomial Derivatives")
ax.grid()
ax.legend()
plt.savefig("legendre_polynomial_derivatives.png")

