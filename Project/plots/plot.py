from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
    x = []
    y1 = []
    y2 = []
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            s = line.split()
            x.append(float(s[0]))
            y1.append(float(s[1]))
            y2.append(float(s[2]))

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.savefig(sys.argv[2])