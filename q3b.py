import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Load the diabetes dataset
import random

x1 = []
x2 = []
# 2.循环100次
for i in range(0, 200):
    # 3.生成随机数
    num = random.randint(0, 50)
    # 4.添加到列表中
    x1.append(num)
print(x1)
for i in range(0, 200):
    # 3.生成随机数
    num = random.randint(0, 50)
    # 4.添加到列表中
    x2.append(num)

x1 = np.array(x1)-20
x2 = np.array(x2)
error1 = np.round(np.random.randn(200), 2)
error2 = np.round(np.random.randn(200), 2)

print(error1)
y1 = -1.5 * x1 + error1 * 20
y2 = -0.8 * x2 + error2 * 20 +90

#
# plt.scatter(x1, y1, color='orange')
# plt.scatter(x2, y2, color='pink')
# plt.xticks(())
# plt.yticks(())
# plt.show()


data2_X = x1
data2_y = y1
data3_X = x2
data3_y = y2

xxx111=list(x1)
xxx222=list(x2)
x=xxx111.extend(xxx222)

yyy1=list(y1)
yyy2=list(y2)
y=yyy1.extend(yyy2)
data1_X = xxx111
data1_y = yyy1

print("000000")
print(data1_X)
print(data1_y)




data1_X = np.array(data1_X).reshape(-1, 1)
data1_y = np.array(data1_y).reshape(-1, 1)
data2_X = np.array(data2_X).reshape(-1, 1)
data2_y = np.array(data2_y).reshape(-1, 1)
data3_X = np.array(data3_X).reshape(-1, 1)
data3_y = np.array(data3_y).reshape(-1, 1)


print(data1_X.shape)
print(data2_X.shape)
print(data3_X.shape)
print(data1_X)
print(data2_X)
print(data3_X)




# size_data=int(len(data_X)*0.2)
size_data1 = int(len(data1_X) * 0.2)
size_data2 = int(len(data2_X) * 0.2)
size_data3 = int(len(data3_X) * 0.2)

data1_X_train = np.array(data1_X[:-size_data1]).reshape(-1, 1)
data1_X_test = np.array(data1_X[-size_data1:]).reshape(-1, 1)

# Split the targets into training/testing sets
data1_y_train = np.array(data1_y[:-size_data1]).reshape(-1, 1)
data1_y_test = np.array(data1_y[-size_data1:]).reshape(-1, 1)

data2_X_train = np.array(data2_X[:-size_data2]).reshape(-1, 1)
data2_X_test = np.array(data2_X[-size_data2:]).reshape(-1, 1)

# Split the targets into training/testing sets
data2_y_train = np.array(data2_y[:-size_data2]).reshape(-1, 1)
data2_y_test = np.array(data2_y[-size_data2:]).reshape(-1, 1)

data3_X_train = np.array(data3_X[:-size_data2]).reshape(-1, 1)
data3_X_test = np.array(data3_X[-size_data2:]).reshape(-1, 1)

# Split the targets into training/testing sets
data3_y_train = np.array(data3_y[:-size_data2]).reshape(-1, 1)
data3_y_test = np.array(data3_y[-size_data2:]).reshape(-1, 1)

# Create linear regression object
regr = linear_model.LinearRegression()
regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
regr3 = linear_model.LinearRegression()

# Train the model using the training sets
# regr.fit(data_X_train, data_y_train)
regr1.fit(data1_X_train, data1_y_train)
regr2.fit(data2_X_train, data2_y_train)
regr3.fit(data3_X_train, data3_y_train)

# Make predictions using the testing set
# data_y_pred = regr.predict(data_X_test)
data_y_pred1 = regr1.predict(data1_X_test)
data_y_pred2 = regr2.predict(data2_X_test)
data_y_pred3 = regr3.predict(data3_X_test)

# The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(data_y_test, data_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(data_y_test, data_y_pred))

# Plot outputs


# p1 = [160, 240] #点p1的坐标值
# p2 = [40, 160] #点p2的坐标值
# plt.plot(p1,p2)

# plt.scatter(data_X_test, data_y_test,  color='black')


# plt.scatter(data1_X_test, data1_y_test, color='black')
plt.plot(data1_X_test, data_y_pred1, color='yellow', linewidth=3)

plt.scatter(data2_X_test, data2_y_test, color='red')
plt.plot(data2_X_test, data_y_pred2, color='pink', linewidth=3)

plt.scatter(data3_X_test, data3_y_test, color='orange')
plt.plot(data3_X_test, data_y_pred3, color='mediumaquamarine', linewidth=3)

#
# plt.plot(data_X_test, data_y_pred, color='blue', linewidth=3)


# plt.xticks(())
# plt.yticks(())
plt.show()
