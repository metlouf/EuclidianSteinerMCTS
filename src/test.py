import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
line, = ax.plot(x, y)
while True:
    user_input = input("Enter new frequency (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    try:
        freq = float(user_input)
        y = np.sin(freq * x)  # Modify the plot dynamically
        ax.clear()
        ax.plot(x, y)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
    except ValueError:
        print("Invalid input, please enter a number.")

plt.ioff()  # Turn off interactive mode

