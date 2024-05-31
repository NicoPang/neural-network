import ml.network as network
import support.functions as functions
import support.processing as processing

if __name__ == '__main__':

    model = network.BasicNeuralNetworkModel([10, 4, 4, 2], seed = 1)
    processing.initialize(processing.xavier_initialize, model)

    model.print_weights()
    model.print_biases()

    node_values = processing.forward_propogate(model, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    print("raw values:")
    for a in node_values:
        print(a)

    shifts = processing.back_propogate(model, node_values, [1, 1, 1])

    print("Weight changes:")
    for i in range(len(shifts[0])):
        print(shifts[0][i])
    print("Bias changes:")
    for i in range(len(shifts[0])):
        print(shifts[1][i])

    model.adjust_weights_biases(shifts[0], shifts[1])

    model.print_weights()
    model.print_biases()