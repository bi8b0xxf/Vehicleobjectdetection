import matplotlib.pyplot as plt


x = []
y = []
for i in range(10):
    x.append(i)
    y.append(i*i)
print(x)

plt.plot(x, y, 'k.-', label="aaa")
plt.imshow()
plt.show()
