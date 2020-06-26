import matplotlib.pyplot as plt
import numpy as np


labels = ['']
CORAL_means = [37.9]
DAN_means = [43.5]
DCORAL_means = [43.2]
DHN_means = [45.5]
DLRC_means = [42.3]
DTLC_means = [47.1]
OUR_means = [62.1]

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x , CORAL_means, width, label='CORAL')
rects2 = ax.bar(x + width, DAN_means, width, label='DAN')
rects3 = ax.bar(x + 2*width, DCORAL_means, width, label='DCORAL')
rects4 = ax.bar(x + 3*width, DHN_means, width, label='DHN')
rects5 = ax.bar(x + 4*width, DLRC_means, width, label='DLRC')
rects6 = ax.bar(x + 5*width, DTLC_means, width, label='DTLC')
rects7 = ax.bar(x + 6*width, OUR_means, width, label='OUR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean of accuracy')
#ax.set_title('Mean of accuracy for different methods on Office31 dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="lower right")


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
autolabel(rects7)

fig.tight_layout()

plt.show()