import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix


class EnsemblePerceptronStep:
    def __init__(self, num_perceptrons):
        self.perceptrons = [Perceptron(random_state=0) for _ in range(num_perceptrons)]

    def predict(self, xs):
        predictions = []
        for x in xs:
            prediction = round(np.mean(np.array([perceptron.predict([x]) for perceptron in self.perceptrons])))
            predictions.append(prediction)
        return predictions

    def train(self, xs, ys):
        for perceptron in self.perceptrons:
            perceptron.fit(xs, ys)


class EnsemblePerceptronSigmoid:
    def __init__(self, num_perceptrons):
        self.perceptrons = MLPClassifier(hidden_layer_sizes=[num_perceptrons], activation='logistic', solver="adam")

    def predict(self, xs):
        predictions = self.perceptrons.predict(xs)
        return predictions

    def train(self, xs, ys):
        self.perceptrons.fit(xs, ys)


print("Выберите функцию активации: 1 - ступенчатая  2 - сигмоидальная")
function_choice = str(input())

if function_choice == '1':
    ensemble = EnsemblePerceptronStep(num_perceptrons=1)

elif function_choice == '2':
    ensemble = EnsemblePerceptronSigmoid(num_perceptrons=10)
else:
    print("input error")
    quit()

(digits, labels) = load_digits(return_X_y=True)

ensemble.train(digits, labels)

prediction = ensemble.predict([digits[0]])
print("Прогноз: " + str(prediction) + " верно?: " + str(prediction == labels[0]))

predictions = ensemble.predict(digits)

conf_m = confusion_matrix(labels, predictions)

print(f"Confusion matrix:\n{conf_m}")

plt.show()
