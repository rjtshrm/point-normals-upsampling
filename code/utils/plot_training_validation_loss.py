# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

state = open("../results/step_1/state_1.txt", "a+")

l = state.readlines()

state.seek(0, 0)
l = state.readlines()

t_loss = [float(i.split(", ")[1].split(" ")[-1]) for i in l]

v_loss = [float(i.split(", ")[2].split(" ")[-1][0:-1]) for i in l]

plt.plot(t_loss, 'r--', label='training loss')
plt.plot(v_loss, '-b', label='validation loss')
plt.yticks(np.arange(min(t_loss), max(t_loss)+1, 1))
plt.xlabel("n iteration")
plt.legend(loc='upper left')
plt.title("T Loss vs V Loss")

plt.show()