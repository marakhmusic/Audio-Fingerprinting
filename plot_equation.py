import numpy as np
import matplotlib.pyplot as plt

def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.show()


y = 'np.sin(x/np.pi) + np.sin(2*x/np.pi) + np.sin(3*x/np.pi)'
graph(y,range(-100,100))
