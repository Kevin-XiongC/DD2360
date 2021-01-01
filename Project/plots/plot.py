from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
    d = sys.argv[1]     # device
    fn = "./plots/perceptron_" + d

    x = []
    y1 = []
    y2 = []
    with open(fn + ".txt", 'r') as f:
        for line in f.readlines():
            s = line.split()
            x.append(float(s[0]))
            y1.append(float(s[1]))
            y2.append(float(s[2]))

    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Cost function")
    plt.legend()

    plt.savefig(fn + ".png")