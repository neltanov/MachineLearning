import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def step_function(t):
    return 1 if t >= 0 else 0


class PerceptronStep:
    def __init__(self, x_size):
        self.weights = np.zeros(x_size)
        self.bias = 0

    def predict(self, xs):
        activation = np.dot(self.weights, xs) + self.bias
        prediction = step_function(activation)
        return prediction

    def train(self, xs, ys, learning_rate=1, epochs=100):
        for epoch in range(epochs):
            trainingIsDone = True
            for x, y in zip(xs, ys):
                prediction_result = self.predict(x)
                if prediction_result != y:

                    if prediction_result == 1:
                        self.weights -= learning_rate * x
                        self.bias -= learning_rate * 1

                    elif prediction_result == 0:
                        self.weights += learning_rate * x
                        self.bias += learning_rate * 1

                    trainingIsDone = False

            if trainingIsDone:
                break


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class PerceptronSigmoid:
    def __init__(self, x_size):
        self.weights = np.zeros(x_size)
        self.bias = 0

    def predict(self, xs):
        activation = np.dot(self.weights, xs) + self.bias
        prediction = sigmoid(activation)
        return prediction

    def findGradient(self, t):
        return t * (1 - t)

    def train(self, xs, ys, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            isTrained = True
            for x, y in zip(xs, ys):
                prediction = self.predict(x)
                gradient = self.findGradient(prediction)
                delta = prediction - y
                if prediction != y:

                    if gradient == 0:
                        break

                    elif gradient > 0:
                        self.weights -= learning_rate * gradient * x * delta
                        self.bias -= learning_rate * gradient * 1 * delta

                    elif gradient < 0:
                        self.weights += learning_rate * gradient * x * delta
                        self.bias += learning_rate * gradient * 1 * delta

                isTrained = False
            if isTrained:
                break


class EnsembleStep:
    def __init__(self, num_perceptrons, x_size):
        self.perceptrons = [PerceptronStep(x_size) for _ in range(num_perceptrons)]

    def predict(self, xs):
        prediction = np.mean(np.array([perceptron.predict(xs) for perceptron in self.perceptrons]))
        return 1 if prediction > 0.5 else 0

    def train(self, xs, ys, learning_rate=1, epochs=100):
        for perceptron in self.perceptrons:
            perceptron.train(xs[0], ys[0], learning_rate, epochs)


class EnsembleSigmoid:
    def __init__(self, num_perceptrons, x_size):
        self.perceptrons = [PerceptronSigmoid(x_size) for _ in range(num_perceptrons)]

    def predict(self, xs):
        prediction = np.mean(np.array([perceptron.predict(xs) for perceptron in self.perceptrons]))
        return 1 if prediction > 0.5 else 0

    def train(self, xs, ys, learning_rate=1, epochs=100):
        for perceptron in self.perceptrons:
            perceptron.train(xs[0], ys[0], learning_rate, epochs)


sample_size = 500

colors_type1 = np.array([1] * sample_size)  # orange points
colors_type2 = np.array([0] * sample_size)  # blue points

xs = []
ys = []


def set_noise(x, y, noise_level=0.1):
    x_noisy = x + np.random.normal(0, noise_level, size=len(x))
    y_noisy = y + np.random.normal(0, noise_level, size=len(y))
    return x_noisy, y_noisy


def ring():
    # orange points
    theta_outer = np.linspace(0, 2 * np.pi, sample_size)
    radius_outer = np.random.uniform(5, 10, size=sample_size)
    x_outer = radius_outer * np.cos(theta_outer)
    y_outer = radius_outer * np.sin(theta_outer)
    x_outer, y_outer = set_noise(x_outer, y_outer)
    samples_type1 = np.column_stack([x_outer, y_outer])

    # blue points
    cluster_center = np.array([0, 0])
    cluster_radius = 2
    theta_inner = np.linspace(0, 2 * np.pi, sample_size)
    radius_inner = np.random.uniform(0, cluster_radius, size=sample_size)
    x_inner = cluster_center[0] + radius_inner * np.cos(theta_inner)
    y_inner = cluster_center[1] + radius_inner * np.sin(theta_inner)
    x_inner, y_inner = set_noise(x_inner, y_inner)
    samples_type2 = np.column_stack([x_inner, y_inner])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    plt.scatter(samples_type1[:, 0], samples_type1[:, 1], color='orange', label='Type 1 (Ring)')
    plt.scatter(samples_type2[:, 0], samples_type2[:, 1], color='blue', label='Type 2 (Cluster)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Sample')
    plt.legend()
    plt.draw()


def squares():
    # orange points
    side_length = 2
    square1 = np.random.uniform(0, side_length, size=(sample_size // 2, 2)) + np.array([0, 1 * side_length])
    square2 = np.random.uniform(0, side_length, size=(sample_size // 2, 2)) + np.array(
        [1 * side_length, 2 * side_length])
    square1[0], square1[1] = set_noise(square1[0], square1[1])
    square2[0], square2[1] = set_noise(square2[0], square2[1])
    samples_type1 = np.vstack([square1, square2])

    # blue points
    square5 = np.random.uniform(0, side_length, size=(sample_size // 2, 2)) + np.array([1 * side_length])
    square6 = np.random.uniform(0, side_length, size=(sample_size // 2, 2)) + np.array([0, 2 * side_length])
    square5[0], square5[1] = set_noise(square5[0], square5[1])
    square6[0], square6[1] = set_noise(square6[0], square6[1])

    samples_type2 = np.vstack([square5, square6])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    plt.scatter(samples_type1[:, 0], samples_type1[:, 1], color='orange', label='Type 1 (Orange Squares)')
    plt.scatter(samples_type2[:, 0], samples_type2[:, 1], color='blue', label='Type 2 (Blue Squares)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.legend()
    plt.draw()


def piles():
    # orange points
    cluster1 = np.random.normal([0, 0], 1, size=(500, 2))
    cluster1[0], cluster1[1] = set_noise(cluster1[0], cluster1[1])

    # blue points
    cluster2 = np.random.normal([4, 4], 1, size=(500, 2))
    cluster2[0], cluster2[1] = set_noise(cluster2[0], cluster2[1])

    samples_type3 = np.vstack([cluster1, cluster2])

    xs.append(samples_type3)
    ys.append([1] * len(cluster1) + [0] * len(cluster2))

    plt.scatter(samples_type3[:, 0], samples_type3[:, 1],
                c=['orange' if i < sample_size else 'blue' for i in range(sample_size * 2)])

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.draw()


def spirals():
    # blue points
    theta_outer = np.linspace(0, 6 * np.pi, sample_size)
    radius_outer = np.linspace(0, 2.8, sample_size)
    x_outer = radius_outer * np.cos(theta_outer)
    y_outer = radius_outer * np.sin(theta_outer)
    x_outer, y_outer = set_noise(x_outer, y_outer)
    samples_type1 = np.column_stack([x_outer, y_outer])

    # orange points
    theta_inner = np.linspace(0, 6 * np.pi, sample_size)
    radius_inner = np.linspace(0, 2, sample_size)
    x_inner = radius_inner * np.cos(theta_inner)
    y_inner = radius_inner * np.sin(theta_inner)
    x_inner, y_inner = set_noise(x_inner, y_inner)
    samples_type2 = np.column_stack([x_inner, y_inner])

    xs.append(np.concatenate([samples_type1, samples_type2]))
    ys.append(np.concatenate([colors_type1, colors_type2]))

    plt.scatter(x_outer, y_outer, c='blue', label='Type 1 (Outer Spiral)')
    plt.scatter(x_inner, y_inner, c='orange', label='Type 2 (Inner Spiral)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Generated Samples')
    plt.legend()
    plt.draw()


np.random.seed(51)

print("Choose sample type: 1 - ring with cluster  2 - squares  3 - piles  4 - spirals")
xs_choice = str(input())

if xs_choice == '1':
    ring()
elif xs_choice == '2':
    squares()
elif xs_choice == '3':
    piles()
elif xs_choice == '4':
    spirals()
else:
    print("input error")
    quit()

print("Activation function: 1 - step  2 - sigmoid")
function_choice = str(input())

if function_choice == '1':
    model = EnsembleStep(num_perceptrons=10, x_size=2)

elif function_choice == '2':
    model = EnsembleSigmoid(num_perceptrons=20, x_size=2)
else:
    print("input error")
    quit()

model.train(xs, ys)

xs = xs[0]
ys = ys[0]

plt.show()

prediction = model.predict(xs[0])
print("Prediction: " + str(prediction) + " is correct: " + str(prediction == ys[0]))

predictions = [model.predict(x) for x in xs]
c_matrix = confusion_matrix(ys, predictions, normalize='true')

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix)

confusion_matrix_display.plot()
plt.show()
