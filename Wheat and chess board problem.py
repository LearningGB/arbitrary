import numpy as np
import matplotlib.pyplot as plt

def wheat_on_chessboard(n, m):

    chessboard = np.zeros((n, m), dtype=np.uint64)


    for i in range(n):
        for j in range(m):
            chessboard[i, j] = 2**(i + j)

    return chessboard

def wheat_on_chessboard_append(n, m):

    chessboard = np.array([1], dtype=np.uint64)


    for _ in range(n * m - 1):
        chessboard = np.append(chessboard, 2 * chessboard[-1])


    chessboard = chessboard.reshape((n, m))

    return chessboard

def wheat_on_chessboard_broadcast(n, m):

    chessboard = np.arange(n * m, dtype=np.uint64)


    chessboard = 2**chessboard


    chessboard = chessboard.reshape((n, m))

    return chessboard


n = 2
m = 2
chessboard = wheat_on_chessboard(n, m)
print("Number of wheat on a 2x2 chessboard:")
print(chessboard)


n = 8
m = 8
chessboard = wheat_on_chessboard(n, m)
print("Number of wheat on an 8x8 chessboard:")
print(chessboard)


total_wheat = np.sum(wheat_on_chessboard(n, m))
print("Total number of wheat:", total_wheat)

plt.imshow(chessboard, cmap='viridis')
plt.colorbar()
plt.show()

first_half = np.sum(chessboard[0:4, :])
second_half = np.sum(chessboard[4:, :])
ratio = second_half / first_half
print("The second half is", ratio, "times as long as the first half.")

chessboard_append = wheat_on_chessboard_append(n, m)
chessboard_broadcast = wheat_on_chessboard_broadcast(n, m)

method1_time = timeit(method1, number=1000)
method2_time = timeit(method2, number=1000)
method3_time = timeit(method3, number=1000)

print("Method 1 execution time:", method1_time)
print("Method 2 execution time:", method2_time)
print("Method 3 execution time:", method3_time)
