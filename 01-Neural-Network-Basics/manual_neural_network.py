import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# operation
class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        

        for node in input_nodes:
            node.output_nodes.append(self)
            
        
    
    def compute(self):
        pass


class Sigmoid(Operation):
    def __init__(self, z):
        super.__init__([z])
    
    def compute(self, z_val):
        return 1/(1+np.exp(-z_val))


class multiply(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])
    
    def compute(self, a_var, b_var):
        self.inputs = [a_var, b_var]
        return a_var * b_var

class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class Placeholder():
    def __init__(self):
        self.output_nodes = []

        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value = None):
        self.value = initial_value
        self.output_nodes = []
    
        _default_graph.variables.append(self)

        

    
class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):
        global _default_graph
        _default_graph = self


class Session():
    def run(self, operation, feed_dict = {}):

        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
                

            elif type(node) == Variable:
                node.output = node.value
                
            else: # Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
                
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output



def traverse_postorder(operation):
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    
    recurse(operation)
    return nodes_postorder


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def main():

    '''
    A Basic Graph 
    With A=10 and b=1
    𝑧=𝐴𝑥+𝑏 --> 𝑧=10𝑥+1
    '''

    g = Graph()
    g.set_as_default()
    
    A = Variable(10)
    b = Variable(1)
    

    

    x = Placeholder()
    y = multiply(A, x)
    

    z = add(y, b)

    sess = Session()
    result = sess.run(operation=z, feed_dict={x:10})
    

    sample_z = np.linspace(-10, 10, 100)
    sample_a = sigmoid(sample_z)



    plt.plot(sample_z, sample_a)
    plt.show()


    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
    features = data[0]
    labels = data[1]
    plt.scatter(features[:,0], features[:,1],c=labels,cmap='coolwarm')
    plt.show()

    

    



if __name__ == '__main__':
    main()