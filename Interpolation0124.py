import matplotlib.pyplot as plt
import numpy as np

# SPLINES FOR PLOTTING DIGIT "0"
#cubic spline 1:
cubic1_x1_0 = np.arange(1, 2, 0.001)
cubic1_S1_0 = 3 - 3*(cubic1_x1_0 - 1) + ((cubic1_x1_0 - 1)**3)

cubic1_x2_0 = np.arange(2, 3, 0.001)
cubic1_S2_0 = 1 + 3*((cubic1_x2_0 - 2)**2) - 1*((cubic1_x2_0 - 2)**3)

#cubic spline 2:
cubic2_x1_0 = np.arange(1, 2, 0.001)
cubic2_S1_0 = 3 + 3*(cubic2_x1_0 - 1) - ((cubic2_x1_0 - 1)**3)

cubic2_x2_0 = np.arange(2, 3, 0.001)
cubic2_S2_0 = 5 - 3*((cubic2_x2_0 - 2)**2) + ((cubic2_x2_0 - 2)**3)


# SPLINES FOR PLOTTING DIGIT "1"
#linear spline 1:
linear1_x1_1 = np.arange(6, 6.1, 0.001)
linear1_S1_1 = 5 - 40*(linear1_x1_1 - 6)

#linear spline 2:
linear2_x1_1 = np.arange(5, 6, 0.001)
linear2_S1_1 = 2*(linear2_x1_1) - 7


# SPLINES PLOTTING DIGIT "2"
#cubic spline 1:
cubic1_x1_2 = np.arange(8, 9, 0.001)
cubic1_S1_2 = 4 + (3/2)*(cubic1_x1_2 - 8) - (1/2)*((cubic1_x1_2 - 8)**3)

cubic1_x2_2 = np.arange(9, 10, 0.001)
cubic1_S2_2 = 5 - (3/2)*((cubic1_x2_2 - 9)**2) + (1/2)*((cubic1_x2_2 - 9)**3)

# linear spline 1:
linear1_x1_2 = np.arange(8, 10, 0.001)
linear1_S1_2 = 4 + (3/2)*(linear1_x1_2 - 10)

# linear spline 2:
linear2_x1_2 = np.arange(8, 10.5, 0.001)
linear2_S1_2 = 1 + 0*(linear2_x1_2 - 8)


# SPLINES PLOTTING DIGIT "4"
#linear spline 1:
linear1_x1_4 = np.arange(12, 15, 0.001)
linear1_S1_4 = 2 + 0*(linear1_x1_4 - 12)

#linear spline 2:
linear2_x1_4 = np.arange(12, 14, 0.001)
linear2_S1_4 = 5 + (3/2)*(linear2_x1_4 - 14)

#linear spline 3:
linear3_x1_4 = np.arange(14, 14.1, 0.001)
linear3_S1_4 = 5 - 40*(linear3_x1_4 -14)



#  PLOTTING 

# plotting 0
plt.plot(cubic1_x1_0,cubic1_S1_0)
plt.plot(cubic2_x2_0,cubic1_S2_0)

plt.plot(cubic2_x1_0, cubic2_S1_0)
plt.plot(cubic2_x2_0, cubic2_S2_0)

# plotting 1
plt.plot(linear1_x1_1, linear1_S1_1)

plt.plot(linear2_x1_1, linear2_S1_1)

# plotting 2
plt.plot(linear1_x1_2, linear1_S1_2)

plt.plot(linear2_x1_2, linear2_S1_2)

plt.plot(cubic1_x1_2, cubic1_S1_2)
plt.plot(cubic1_x2_2, cubic1_S2_2)

# plotting 4
plt.plot(linear1_x1_4, linear1_S1_4)

plt.plot(linear2_x1_4, linear2_S1_4)

plt.plot(linear3_x1_4, linear3_S1_4)


plt.axis((0,20,0,15))
plt.show()
