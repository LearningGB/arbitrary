import matplotlib.pyplot as plt
import math

chestnut_bun_volume = 523.6  # cubic centimeters


solar_system_volume = 1.41e27  # cubic kilometers


number_of_chestnut_buns = solar_system_volume / chestnut_bun_volume


byvine_doubling_time = 5  # minutes


time_to_cover_solar_system = 0
number_of_chestnut_buns_list = []
time_list = []

for i in range(int(math.log2(number_of_chestnut_buns))):
    time_to_cover_solar_system += byvine_doubling_time
    number_of_chestnut_buns_list.append(number_of_chestnut_buns)
    time_list.append(time_to_cover_solar_system)

    number_of_chestnut_buns = number_of_chestnut_buns * 2

plt.plot(time_list, number_of_chestnut_buns_list)
plt.xlabel("Time (minutes)")
plt.ylabel("Number of chestnut buns")
plt.title("Number of chestnut buns vs. time")
plt.show()
