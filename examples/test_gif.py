import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
my_plot, = plt.plot([], [], "ro", animated=True)

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    xdata = [random.randint(0, 10)]
    ydata = [random.randint(0, 10)]
    my_plot.set_data(xdata, ydata)
    return my_plot,

def update(frame):
    xdata = [random.randint(0, 10)]
    ydata = [random.randint(0, 10)]
    my_plot.set_data(xdata, ydata)
    return my_plot,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 10),
                    init_func=init, interval=1000, blit=True)
ani.save("test.gif")
plt.show()