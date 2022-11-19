import random
import matplotlib.pyplot as plt

# True values
x1 = [-1, -1, 1, 1]
x2 = [-1, 1, -1, 1]
y = [-1, -1, -1, 1]

# Set random weights (0< <1)
w1 = random.random()
w2 = random.random()
b = random.random()

# Learning rule
eita = 0.5


# Activation function
def af(a):
    if a >= 0:
        return 1
    else:
        return -1


# Training process
while True:
    errors = 0

    for index in range(4):
        net = x1[index] * w1 + x2[index] * w2 + b
        predict = af(net)

        if predict != y[index]:
            errors += 1

        # Delta W calculation
        delta_w1 = eita * (y[index] - predict) * x1[index]
        delta_w2 = eita * (y[index] - predict) * x2[index]
        delta_b = eita * (y[index] - predict)

        # New weights
        w1 += delta_w1
        w2 += delta_w2
        b += delta_b

    # print(errors)
    if errors == 0:
        break

print(f'w1={w1}\nw2={w2}\nb={b}')

# Plot

#Labels
plt.xlabel(f'X1\nw1: {w1}\nw2: {w2}\nb: {b}')
plt.ylabel('X2')

# Plot line --> Using 2 points of the line --> (2, y1), (-2, y2)
y1 = -(2 * w1 + b) / w2
y2 = -(-2 * w1 + b) / w2

plt.plot([2, -2], [y1, y2])
plt.plot([-1, -1, 1], [-1, 1, -1], 'ro')
plt.plot([1], [1], 'bo')

plt.show()
