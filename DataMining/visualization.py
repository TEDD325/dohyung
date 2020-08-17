import random
import matplotlib.pyplot as plt
# from matplotlib.backends.qt_editor.figureoptions import LINESTYLES

# TODO: please add your code here
random.seed(0)
x = range(5)
y = [random.randint(10, 20) 
     for __ in range(len(x))]

plt.plot(x, y, color='green', marker='o', linestyle='solid')  # add a polyline
plt.plot(x, [random.randint(10, 20)
             for _ in range(len(x))],
             color=(random.random(), random.random(), random.random()),
             marker='D', linestyle=':')
plt.plot(y, [random.randint(10, 20)
             for _ in range(len(x))],
             color=(random.random(), random.random(), random.random()),
             marker='>', linestyle='--')
plt.title("Sample")  # add a title
plt.xlabel("Attribute 1")  # set the x-axis label
plt.ylabel("Attribute 2")  # set the y-axis label
plt.show()  # show the figure

plt.scatter(x, y, color='red', marker='x')  # add a scatter plot
plt.xlabel("Attribute 1")  # set the x-axis label
plt.ylabel("Attribute 2")  # set the y-axis label
plt.show()  # show the figure

labels = ["A", "B", "C", "D", "E"]  # custom labels
plt.bar(x, y)  # plot bars
plt.xticks(x, labels)  # add custom labels below each bar
plt.xlabel("Attribute 1")  # set the x-axis label
plt.ylabel("Attribute 2")  # set the y-axis label
plt.show()  # show the figure

plt.pie([0.95, 0.05], labels=["Happy", "Unhappy"]) # draw a pie chart
plt.show()  # show the figure