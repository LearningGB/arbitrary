import time
import matplotlib.pyplot as plt

THICKNESS = 0.00008


start = time.time()
exponentiation_folded_thickness = THICKNESS * 2 ** 43
elapsed_time_exponentiation = time.time() - start


start = time.time()
for_folded_thickness = THICKNESS
for _ in range(43):
    for_folded_thickness *= 2
elapsed_time_for = time.time() - start


folded_thickness_list = []


folded_thickness = THICKNESS
for i in range(43):
    folded_thickness = folded_thickness*2
    folded_thickness_list.append(folded_thickness)


print("Execution time with exponentiation arithmetic operators:", elapsed_time_exponentiation, "seconds")
print("Execution time with for statement:", elapsed_time_for, "seconds")


plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness [m]")
plt.plot(folded_thickness_list, color='red', linewidth=2, linestyle='--')
plt.tick_params(labelsize=20)
plt.show()


print("Thickness of the paper after 43 folds:", folded_thickness_list[-1], "meters")


print("Thickness in kilometers:", "{: .2f}".format(folded_thickness_list[-1] / 1000), "kilometers")


distance_to_moon = 384400
if folded_thickness_list[-1] > distance_to_moon:
    print("The folded paper reaches the moon!")
else:
    print("The folded paper does not reach the moon.")

print("The thickness of the paper increases exponentially with the number of folds. This means that the thickness increases rapidly as the number of folds increases.")
