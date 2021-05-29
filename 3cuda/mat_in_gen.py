import random

n = 1000
with open('in.txt', 'w') as f:
    for i in range(n):
        for j in range(n):
            print(random.random(), file=f)
    for i in range(n):
        for j in range(n):
            print(random.random(), file=f)
