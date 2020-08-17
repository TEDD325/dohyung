import math, random


def construct_network(input_size,  # the number of input features
      output_size,  # the number of output features
      hidden_layer_size):  # the number of neurons in the hidden layer
    hidden_layer = [[random.random()  # hidden layer
        for __ in range(input_size + 1)]  # one weight per input feature and a bias weight
        for __ in range(hidden_layer_size)]  # for each hidden neuron
    output_layer = [[random.random()  # output layer
        for __ in range(hidden_layer_size + 1)]  # one weight per input feature and a bias weight
        for __ in range(output_size)]  # for each output neuron
    return [hidden_layer, output_layer]


def predict(network, example):
    return feed_forward(network, example)[-1]


def feed_forward(network, input_vector):
    results = []
    for layer in network:  # for each layer
        input_with_bias = input_vector + [1]  # add a bias input
        result = [output(neuron, input_with_bias)  # result from this layer
                  for neuron in layer]
        results.append(result)  # register the result
        input_vector = result  # result from this layer = input to the next layer
    return results


def output(weights, input_vector):
    return sigmoid(dot_product(weights, input_vector))


def sigmoid(t):
    return 1 / (1 + math.exp(-t))


def backpropagate(network, example, target):
    hidden_result, result = feed_forward(network, example)
    # result_i * (1 - result_i) from the derivative of sigmoid
    result_delta = [result_i * (1 - result_i) * (result_i - target[i])
                     for i, result_i in enumerate(result)]
    # adjust weights in the output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_result_j in enumerate(hidden_result + [1]):
            output_neuron[j] -= result_delta[i] * hidden_result_j
    # back-propagate errors to the hidden layer
    hidden_deltas = [hidden_result_j * (1 - hidden_result_j) * 
                      dot_product(result_delta, [n[i] for n in network[-1]])
                     for i, hidden_result_j in enumerate(hidden_result)]
    # adjust weights in the hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, e in enumerate(example + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * e


def dot_product(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def vectorize(string):
    return [1 if c == '@' else 0
            for row in string.split("\n")
            for c in row.strip()]

'''
def show_weights(neuron_idx):
    weights = network[0][neuron_idx]
    abs_weights = [abs(weight) for weight in weights]
    grid = [abs_weights[row:(row + 5)]
            for row in range(0, 25, 5)]
    ax = plt.gca()
    ax.imshow(grid,
#              cmap=plt.cm.hot,
              interpolation='none')
    for i in range(5):
        for j in range(5):
            if weights[5 * i + j] < 0:
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
    plt.show()


def patch(x, y, hatch, color):
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def vectorize(string):
    return [1 if c == '1' else 0
            for row in string.split("\n")
            for c in row.strip()]
'''

if __name__ == "__main__":

    training_data = [
          """@@@@@
             @...@
             @...@
             @...@
             @@@@@""",

          """..@..
             ..@..
             ..@..
             ..@..
             ..@..""",

          """@@@@@
             ....@
             @@@@@
             @....
             @@@@@""",

          """@@@@@
             ....@
             @@@@@
             ....@
             @@@@@"""]
       
    inputs = list(map(vectorize, training_data))
    for string in training_data:
        print(string.replace(" ", ""))
        print()

    targets = [[1 if i == j else 0 for i in range(len(training_data))]
               for j in range(len(training_data))]

    random.seed(0)
    
    # construct a network having 5 neurons in its hidden layer
    # for input vectors of length 25 and output vectors of length 10
    network = construct_network(25, len(training_data), 5)
    
    traning_number = 10000
    for __ in range(traning_number):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    for i, input_vector in enumerate(inputs):
        results = predict(network, input_vector)
        print(i, [round(p, 2) for p in results])
    
    test_data = [
          """@@@@.
             @..@.
             @..@@
             @..@@
             @@@@.""",

          """...@
             ...@.
             ...@.
             ..@@.
             ..@@.""",

          """@@@@.
             ...@.
             @@@@.
             @....
             @@@@@""",

          """@@@@.
             ....@
             @@@@.
             ....@
             @@@@.""",

          """.@@@.
             @...@
             .@@@.
             @...@
             .@@@."""]

    for example in test_data:
        print(example.replace(" ", ""))
        print([round(x, 2) for x in
              predict(network, vectorize(example))])
        print()
