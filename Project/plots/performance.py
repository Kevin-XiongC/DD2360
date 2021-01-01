''' data
epoch 40
batch size 100
learning rate 0.001
lambda 0.3

CPU with cost
time	accuracy
107528	0.340
109214	0.347
107970	0.344
109994	0.354
109368	0.345

GPU (10, 10) with cost
time	accuracy
70874   0.349
70025   0.341
70346   0.35
70353   0.35
70352   0.343

GPU (10, 20) with cost
time	accuracy
49947   0.345
49431   0.345
51531   0.343
51115   0.340
49820   0.343

GPU (10, 50) with cost
time	accuracy
31984	0.34
32066	0.341
33078   0.336
31596   0.344
32185   0.339

GPU (10, 100) with cost
time	accuracy
28786	0.338
28028	0.341
28089	0.347
27685	0.343
27870	0.342

--------------------

CPU without cost
time	accuracy
70074   0.345
68301   0.354
68912   0.346
69376   0.342
70933   0.349

GPU (10, 10) without cost
time	accuracy
52572   0.349
53019   0.345
53055   0.342
52192   0.345
52725   0.335

GPU (10, 20) without cost
time	accuracy
38098   0.336
38077   0.337
37214   0.343
37446   0.343
38207   0.345

GPU (10, 50) without cost
time	accuracy
27730   0.349
27999   0.348
28092   0.339
27945   0.339
27988   0.346

GPU (10, 100) without cost
time	accuracy
25934   0.337
24805   0.338
24672   0.344
24929   0.344
25980   0.343
'''

import matplotlib.pyplot as plt
import numpy as np

labels = ['CPU', 'GPU (10, 10)', 'GPU (10, 20)', 'GPU (10, 50)', 'GPU (10, 100)']

time_cost = [108.8148, 70.3846, 50.3688, 32.1818, 28.0916]
time_nocost = [69.6532, 54.7126, 37.8084, 27.9508, 25.264]
accuracy_cost = [0.346, 0.3466, 0.3432, 0.34, 0.3422]
accuracy_nocost = [0.3472, 0.3432, 0.3408, 0.3442, 0.3412]

x = np.arange(len(labels))
width = 0.3
fig, ax = plt.subplots()
ax.bar(x - width / 2, time_cost, width, label='with cost computation')
ax.bar(x + width / 2, time_nocost, width, label='without cost computation')

ax.set_ylabel('Average Executing Time (s)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('plots/performance.png')

# fig, ax = plt.subplots()
# ax.bar(x, accuracy_cost, width)
# ax.set_ylabel('Accuracy on test set')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# plt.savefig('plots/accuracy.png')