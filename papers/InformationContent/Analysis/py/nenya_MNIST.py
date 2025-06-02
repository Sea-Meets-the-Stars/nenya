""" Run Nenya on the MNIST dataset """

from nenya.train import main as train_main

def main():

    # Train the model
    train_main("opts_nenya_mnist.json", debug=False)

# Command line execution
if __name__ == '__main__':
    main()