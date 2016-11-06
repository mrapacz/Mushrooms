import matplotlib.pyplot as plt


def plot_results(args, values):
    plt.plot(args, values)

    plt.xlabel('Fraction of data used to train Logistic Regression')
    plt.ylabel('Result efficacy')

    plt.show()
