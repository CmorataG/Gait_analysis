import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import norm
import math

df_results = pd.read_excel("Results_2.xlsx", header=1, usecols=lambda x: 'Unnamed' not in x,)

# Plot predicted vs true values

fig = plt.figure()

# Step rate
ax1 = fig.add_subplot(2,3,1)
x = df_results["Step rate"]
y = df_results["Sr est."]
ax1.scatter(x, y)
p1 = max(max(y), max(x))
p2 = min(min(y), min(x))
ax1.plot([p1, p2], [p1, p2], c="k")
ax1.set_xlabel("True Value")
ax1.set_ylabel("Predicted value")
ax1.set_title("Step rate")
ax1.grid()

# Contact time L
ax2 = fig.add_subplot(2,3,2)
x = df_results["Contact time L"]
y = df_results["Ctl est."]
ax2.scatter(x, y)
p1 = max(max(y), max(x))
p2 = min(min(y), min(x))
ax2.plot([p1, p2], [p1, p2], c="k")
ax2.set_xlabel("True Value")
ax2.set_ylabel("Predicted value")
ax2.set_title("Contact time L")
ax2.grid()

# Contact time R
ax3 = fig.add_subplot(2,3,3)
x = df_results["Contact time R"]
y = df_results["Ctr est."]
ax3.scatter(x, y)
p1 = max(max(y), max(x))
p2 = min(min(y), min(x))
ax3.plot([p1, p2], [p1, p2], c="k")
ax3.set_xlabel("True Value")
ax3.set_ylabel("Predicted value")
ax3.set_title("Contact time R")
ax3.grid()

# Flight Ratio L
ax4 = fig.add_subplot(2,3,4)
x = df_results["Flight Ratio L"]
y = df_results["Frl est."]
ax4.scatter(x, y)
p1 = max(max(y), max(x))
p2 = min(min(y), min(x))
ax4.plot([p1, p2], [p1, p2], c="k")
ax4.set_xlabel("True Value")
ax4.set_ylabel("Predicted value")
ax4.set_title("Flight Ratio L")
ax4.grid()

# Flight Ratio R
ax6 = fig.add_subplot(2,3,5)
x = df_results["Flight Ratio R"]
y = df_results["Frr est."]
ax6.scatter(x, y)
p1 = max(max(y), max(x))
p2 = min(min(y), min(x))
ax6.plot([p1, p2], [p1, p2], c="k")
ax6.set_xlabel("True Value")
ax6.set_ylabel("Predicted value")
ax6.set_title("Flight Ratio R")
ax6.grid()

fig.tight_layout()
plt.show()


# Errors

df_errors = df_results[["Vídeo"]]
df_errors = df_errors.assign(Step_rate = df_results["Sr est."] - df_results["Step rate"])
df_errors =  df_errors.assign(Contact_time_L = df_results["Ctl est."] - df_results["Contact time L"])
df_errors =  df_errors.assign(Contact_time_R = df_results["Ctr est."] - df_results["Contact time R"])
df_errors =  df_errors.assign(Flight_ratio_L = df_results["Frl est."] - df_results["Flight Ratio L"])
df_errors =  df_errors.assign(Flight_ratio_R = df_results["Frr est."] - df_results["Flight Ratio R"])

# Plot errors

fig = plt.figure()

# Step rate
ax1 = fig.add_subplot(2,3,1)
x = df_errors["Step_rate"]
ax1.hist(x)
mu = np.mean(x)
sigma = math.sqrt(np.var(x))
x_n = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax1.plot(x_n, len(x)*norm.pdf(x_n, mu, sigma))
ax1.set_xlabel("Error")
ax1.set_title("Step rate")

# Contact time L
ax2 = fig.add_subplot(2,3,2)
x = df_errors["Contact_time_L"]
ax2.hist(x)
mu = np.mean(x)
sigma = math.sqrt(np.var(x))
x_n = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax2.plot(x_n, len(x)*norm.pdf(x_n, mu, sigma))
ax2.set_xlabel("Error")
ax2.set_title("Contact time L")

# Contact time R
ax3 = fig.add_subplot(2,3,3)
x = df_errors["Contact_time_R"]
ax3.hist(x)
mu = np.mean(x)
sigma = math.sqrt(np.var(x))
x_n = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax3.plot(x_n, len(x)*norm.pdf(x_n, mu, sigma))
ax3.set_xlabel("Error")
ax3.set_title("Contact time R")

# Flight ratio L
ax4 = fig.add_subplot(2,3,4)
x = df_errors["Flight_ratio_L"]
ax4.hist(x)
mu = np.mean(x)
sigma = math.sqrt(np.var(x))
x_n = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax4.plot(x_n, len(x)*norm.pdf(x_n, mu, sigma))
ax4.set_xlabel("Error")
ax4.set_title("Flight ratio L")

# Flight ratio R
ax5 = fig.add_subplot(2,3,5)
x = df_errors["Flight_ratio_R"]
ax5.hist(x)
mu = np.mean(x)
sigma = math.sqrt(np.var(x))
x_n = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax5.plot(x_n, len(x)*norm.pdf(x_n, mu, sigma))
ax5.set_xlabel("Error")
ax5.set_title("Flight ratio R")

plt.show()


# QQ Plot errors

# Step rate
qqplot(df_errors["Step_rate"], line ='s')
plt.title("Step rate errors")
plt.show()

# Contact time L
qqplot(df_errors["Contact_time_L"], line ='s')
plt.title("Contact time L")
plt.show()

# Contact time R
qqplot(df_errors["Contact_time_R"], line ='s')
plt.title("Contact time R")
plt.show()

# Flight ratio L
qqplot(df_errors["Flight_ratio_L"], line ='s')
plt.title("Flight ratio L")
plt.show()

# Flight ratio R
qqplot(df_errors["Flight_ratio_R"], line ='s')
plt.title("Flight ratio R")
plt.show()


# normality test
stat, p = shapiro(df_errors["Step_rate"])
print("\nStep rate:")
print('Statistics=%.3f, p=%.6f' % (stat, p))

stat, p = shapiro(df_errors["Contact_time_L"])
print("\nContact time L:")
print('Statistics=%.3f, p=%.6f' % (stat, p))

stat, p = shapiro(df_errors["Contact_time_R"])
print("\nContact time R:")
print('Statistics=%.3f, p=%.6f' % (stat, p))

stat, p = shapiro(df_errors["Flight_ratio_L"])
print("\nFlight ratio L:")
print('Statistics=%.3f, p=%.6f' % (stat, p))

stat, p = shapiro(df_errors["Flight_ratio_R"])
print("\nFlight ratio R:")
print('Statistics=%.3f, p=%.6f' % (stat, p))