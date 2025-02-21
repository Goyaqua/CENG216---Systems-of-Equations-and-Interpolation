import numpy as np
import matplotlib.pyplot as plt

# Defining the coordinates of the cities, (x,y)
x = np.array([-2,0,3,5])
y = np.array([3,1,0,2])

def solving_least_square(A_trans_A,A_trans_b):
    #calculate a.transpose().A.XLS = A.transpose().b
    xls = np.linalg.solve(A_trans_A,A_trans_b) 
    return xls # c1,c2, c3

print("Coordinates of the cities are : ")
for xi, yi in zip(x, y):
    print(f"({xi}, {yi})")

# question a)
print()
print("-------Question a-------\n")
print("Linear Fit Calculation y = c1 + c2t\n")

#creating matrix A
matrix_A1 = np.array([[1,x[0]],[1,x[1]],[1,x[2]],[1,x[3]]])
print("A:")
print(matrix_A1 ,"\n")

#creating vector b
b = np.array([y[0],y[1],y[2],y[3]])
print("b:")
print(b, "\n")

print("To solve this equation we need to solve (AT*A) * XLS = ATb equation.\n")

#calculate A.transpose().A
A1_trans = matrix_A1.transpose()
A1_trans_A1 = A1_trans @ matrix_A1
print("transpose of A:")
print(A1_trans,"\n")
print("AT*A : ")
print(A1_trans_A1,"\n")

#calculate A.trasnpose().b
A1_trans_b = A1_trans @ b 
print("AT*b : ")
print(A1_trans_b,"\n")

#calculate a.transpose().A.XLS = A.transpose().b 
c1 = solving_least_square(A1_trans_A1,A1_trans_b)
print("when we solve this equation we get the coefficients, c1 and c2:")
print("c1 = "+str(c1[0])+"\n")
print("c2 = "+str(c1[1])+"\n")


#print the fitting line
print("our fitting line's equation is:")
print("y = " + str(c1[0]) + " + " + str(c1[1]) + "t\n")

#plot the line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='City Coordinates')
t = np.linspace(min(x) - 1, max(x) + 1, 100)  # Generating points for line
y_fit = c1[0] + c1[1] * t  # Calculate y values based on the fit
plt.plot(t, y_fit, 'b-', label=f'Fitting Line: y = {c1[0]:.2f} + {c1[1]:.2f}t')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()

# question b)
print("-------Question b-------\n")
print("Quadratic Fit Calculation y = c1 + c2t + c3t^2\n")

#creating matrix A
matrix_A2 = np.array([[1,x[0],x[0]**2],[1,x[1],x[1]**2],[1,x[2],x[2]**2],[1,x[3],x[3]**2]])
print("A:")
print(matrix_A2,"\n")

print("b:")
print(b,"\n")

print("To solve this equation again, we need to solve (AT*A) * XLS = ATb equation.\n")


#calculate A.transpose().A = A_trans_A
A2_trans = matrix_A2.transpose()
A2_trans_A2 = A2_trans @ matrix_A2
print("transpose of A:")
print(A2_trans,"\n")
print("AT*A : ")
print(A2_trans_A2,"\n")

#calculate A.trasnpose().b = A_trans_b
A2_trans_b = A2_trans @ b 
print("AT*b : ")
print(A2_trans_b,"\n")

#calculate a.transpose().A.XLS = A.transpose().b = least_square()
c2 = solving_least_square(A2_trans_A2,A2_trans_b)
print("when we solve this equation we get the coefficients: c1, c2 and c3 :")
print("c1 = "+str(c2[0])+"\n")
print("c2 = "+str(c2[1])+"\n")
print("c3 = "+str(c2[2])+"\n")

#print the fitting line
print("our fitting line's equation is:")
print("y = " + str(c2[0]) + " + " + str(c2[1]) + "t" + " + " + str(c2[2]) + "t^2\n")

#plot the line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='City Coordinates')
t = np.linspace(min(x) - 1, max(x) + 1, 100)  # Generate values for t to plot the curve
y_fit_quadratic = c2[0] + c2[1] * t + c2[2] * t**2  # Calculate y values based on the quadratic fit
plt.plot(t, y_fit_quadratic, 'blue', label=f'Quadratic Fit: y = {c2[0]:.2f} + {c2[1]:.2f}t + {c2[2]:.2f}t^2')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()

# RMSE Calculation ||r||/ m^(1/2)
print("-------Question c-------\n")
print("To find the root squared error we need to calculate :")
print("||r||/ m^(1/2)\n")

print("For Linear Fitting:\n")

print("find the r = b - A.XLS")
ra = b - matrix_A1 @ c1
print("r is: ")
print(ra,"\n")

print("find the SE = ||r||^2")
squared_errors_a = 0
for i in ra:
    squared_errors_a += i**2
print("SE is: ")
print(squared_errors_a,"\n")

#calculate root mean error
rmse_a = np.sqrt(squared_errors_a / len(b))
print("Since m = 4, our RMSE is:  ")
print(rmse_a,"\n")


print("For Quadratic Fitting:\n")

print("find the r = b - A.XLS again")
rb = b - matrix_A2 @ c2
print("rb is: ")
print(rb,"\n")

print("find the SE = ||r||^2")
squared_errors_b = 0
for i in rb:
    squared_errors_b += i**2
print("SE is:")
print(squared_errors_b)
print("")
#calculate root mean error
rmse_b = np.sqrt(squared_errors_b / len(b))
print("Since m = 4, our RMSE is:  ")
print(rmse_b,"\n")

print("-------Question d-------\n")

# QR Factorization and Solving Least Squares
print("Solving Linear Fitting with QR Factorization :\n")

print("A = Q*R\n")
print("A:")
print(matrix_A1)

a1_a = A1_trans[0]
y1_a = a1_a

#to find normalized y1
squared_y1_a = 0
for i in y1_a:
    squared_y1_a += i**2
normalized_y1_a = np.sqrt(squared_y1_a)

# finding q1_a
q1_a = y1_a / normalized_y1_a

a2_a = A1_trans[1]
q1_trans_a2_a = q1_a.transpose() @ a2_a
y2_a = a2_a - np.dot(q1_trans_a2_a,q1_a)

#to find normalized y2
squared_y2_a = 0
for i in y2_a:
    squared_y2_a += i**2
normalized_y2_a = np.sqrt(squared_y2_a)

# finding q2_a
q2_a = y2_a / normalized_y2_a

#since we found our q vectors we can write Q matrix:
Q_a = np.array([q1_a, q2_a]).transpose()
print("Q : ")
print(Q_a,"\n")

#find R matrix
R_a = np.array([[normalized_y1_a, q1_trans_a2_a], 
                [0,normalized_y2_a]])
print("R :")
print(R_a,"\n")

#solve the least square method of this equation A = QR:
#R * XLS = QT*b
#transpose of Q:
Q_a_trans = Q_a.transpose()
#QT*b:
Q_a_trans_b = Q_a_trans @ b

c_qr_1 = np.linalg.solve(R_a, Q_a_trans_b)
print("When we solve R*XLS = QT*b we get the coefficients: c1, c2:")
print("c1 = "+str(c_qr_1[0])+"\n")
print("c2 = "+str(c_qr_1[1])+"\n")

#print the fitting line
print("Our fitting line's equation is:")
print("y = " + str(c_qr_1[0]) + " + " + str(c_qr_1[1]) + "t")
print()

#plot the line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='City Coordinates')
t = np.linspace(min(x) - 1, max(x) + 1, 100)  # Generating points for line
y_fit_qr1 = c_qr_1[0] + c_qr_1[1] * t  # Calculate y values based on the fit
plt.plot(t, y_fit_qr1, 'b-', label=f'Fitting Line: y = {c_qr_1[0]:.2f} + {c_qr_1[1]:.2f}t')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()

print("Solving Quadratic Fitting with QR Factorization :\n")
print("A:")
print(matrix_A2,"\n")

a1_b = A2_trans[0]
y1_b = a1_b

#to find normalized y1
squared_y1_b = 0
for i in y1_b:
    squared_y1_b += i**2
normalized_y1_b = np.sqrt(squared_y1_b)

# finding q1_b
#q1 = y1/||y1||
q1_b = y1_b / normalized_y1_b

a2_b = A2_trans[1]
#y2 = a2 - q1*(q1t*a2)
q1_trans_a2_b = q1_b.transpose() @ a2_b
y2_b = a2_b - np.dot(q1_trans_a2_b, q1_b)

#to find normalized y2
squared_y2_b = 0
for i in y2_b:
    squared_y2_b += i**2
normalized_y2_b = np.sqrt(squared_y2_b)

# finding q2_a
#q2 = y2/||y2||
q2_b = y2_b / normalized_y2_b

a3_b = A2_trans[2]
#y3 = a3 - q1*(q1t*a3) - q2*(q2t*y)
q1_trans_a3_b = q1_b.transpose() @ a3_b
q2_trans_a3_b = q2_b.transpose() @ a3_b

y3_b = a3_b - np.dot(q1_trans_a3_b,q1_b) - np.dot(q2_trans_a3_b,q2_b)

squared_y3_b = 0
for i in y3_b:
    squared_y3_b += i**2
normalized_y3_b = np.sqrt(squared_y3_b)
#q3 = y3/||y3||
q3_b = y3_b / normalized_y3_b

#to solve with full QR factorization 
#choose a vector that is linearly independent

a4_b = np.array([1,0,0,0])
q1_trans_a4_b = q1_b.transpose() @ a4_b
q2_trans_a4_b = q2_b.transpose() @ a4_b

y4_b = a4_b - np.dot(q1_trans_a4_b, q1_b) - np.dot(q2_trans_a4_b,q2_b)

squared_y4_b = 0
for i in y4_b:
    squared_y4_b += i**2
normalized_y4_b = np.sqrt(squared_y4_b)
#q3 = y3/||y3||
q4_b = y4_b / normalized_y4_b

Q_b = np.array([q1_b,q2_b,q3_b,q4_b ]).transpose()
print("Q : ")
print(Q_b,"\n")

R_b = np.array([[normalized_y1_b, q1_trans_a2_b, q1_trans_a3_b],
                [0, normalized_y2_b, q2_trans_a3_b],
                [0,0, normalized_y3_b],
                [0,0,0]])
print("R :")
print(R_b,"\n")


#solve the least square method of this equation A = QR:
#R * XLS = QT*b
#transpose of Q:
Q_b_trans = Q_b.transpose()
#QT*b:
Q_b_trans_b = Q_b_trans @ b

c_qr_2 = np.linalg.solve(R_b[0:3,:], Q_b_trans_b[0:3])
print("When we solve R*XLS = QT*b we get the coefficients: c1, c2 and c3:")
print("c1 = "+str(c_qr_2[0])+"\n")
print("c2 = "+str(c_qr_2[1])+"\n")
print("c3 = "+str(c_qr_2[2])+"\n")
print()
#print the fitting line
print("our fitting line's equation is:")
print("y = " + str(c_qr_2[0]) + " + " + str(c_qr_2[1]) + "t" + " + " + str(c_qr_2[2]) + "t^2")
print()
# Functions for plotting

#plot the line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='City Coordinates')
t = np.linspace(min(x) - 1, max(x) + 1, 100)  # Generate values for t to plot the curve
y_fit_quadratic_qr2 = c_qr_2[0] + c_qr_2[1] * t + c_qr_2[2] * t**2  # Calculate y values based on the quadratic fit
plt.plot(t, y_fit_quadratic_qr2, 'blue', label=f'Quadratic Fit: y = {c_qr_2[0]:.2f} + {c_qr_2[1]:.2f}t + {c_qr_2[2]:.2f}t^2')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()