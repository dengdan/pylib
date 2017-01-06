import matplotlib.pyplot as plt
import util
def plot_result(path):
    solver = util.io.load(path)
    training_losses = solver.training_losses;
    iterations = len(training_losses);
    plt.plot(range(iterations), training_losses);
    plt.show();
    
    
if __name__== '__main__':
    import argparse
    import os
    import logging	
    parser = argparse.ArgumentParser(description='show the training losses')
    parser.add_argument('--path', type=str, help='the path of solver data to be show')
    args = parser.parse_args().__dict__
    logging.info('**************Arguments*****************')
    logging.info(args)
    logging.info('****************************************')
    plot_result(**args)
