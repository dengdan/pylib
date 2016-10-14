#encoding = utf-8
import logging
import time

import theano
import numpy as np
import theano.tensor as T

import util.io
import util.dtype as dtype
class Solver(object):
    def __init__(self, 
        batch_size, 
        train_iter, 
        epochs = None,
        total_iterations = None,
        learning_rate = 0.01,
        val_iter = None,
        val_interval = 1000,
        dump_interval = 5000,
        dump_path = '.'
        ):
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.batch_size = batch_size
        self.epochs = epochs        
        self.total_iterations = total_iterations
        self.learning_rate = dtype.cast(learning_rate, dtype.floatX)
        self.dump_path = dump_path;
        self.dump_interval = dump_interval
        self.val_interval = val_interval
        
        
        # if total_iterations is set, use it. or calculate it through epochs
        if self.epochs != None and self.total_iterations == None:
            self.total_iterations = self.epochs * self.train_iter.batch_per_epoch
            
        if self.total_iterations == None:
            raise ValueError('Either epochs or total_iterations must be provided. If both are provided, total_iterations has a higher priority.')
    
    def fit(self, model):
        pass
    
    def get_updates(self, model):
        pass
        
    
        
class SimpleGradientDescentSolver(Solver):
    """The most simple gradient descent model, with learning rate fixed after chosen."""
    def __init__(self, 
        train_iter, 
        batch_size, 
        epochs = None,
        total_iterations = None,
        learning_rate = 0.0001,
        val_iter = None,
        val_interval = 1000,
        dump_interval = 5000,
        dump_path = '.'
        ):

        Solver.__init__(self,
            train_iter = train_iter, 
            batch_size = batch_size, 
            epochs = epochs,
            total_iterations = total_iterations,
            val_interval = val_interval,
            dump_interval = dump_interval,
            val_iter = val_iter,
            dump_path = dump_path
            )       
    
    def get_updates(self, model):
        params = model.params_to_be_updated
        updates = [(p, p - self.learning_rate * T.grad(model.loss, p)) for p in params]
        return updates 
        
        
    def fit(self, model):
        training_start = time.time()
        train_iter = self.train_iter
        val_iter = self.val_iter
        logging.info( 'building theano functions...')
        logging.info('building the training function of %s...' %(model.name))
        t1 = time.time()
        training_fn = theano.function(
                inputs = [model.input, model.label], 
                outputs = [model.loss, model.accuracy], 
                updates = self.get_updates(model)
        )
        t2 = time.time()
        logging.info("building finished, using %d seconds."%(t2 - t1))        
        
        
        t1 = time.time() 
        
        if val_iter != None:       
            val_fn = model.get_val_fn()
            
        get_param_count_fn = model.get_param_count_fn()
        t2 = time.time()
        
        logging.info('All functions are built.')

        logging.info("There are %d parameters in %s"%( get_param_count_fn(), model.name))


        training_losses = []
        training_accuracies = []
        val_losses = []
        val_accuracies = []
        dump_path = self.dump_path
        batch_size = self.batch_size
        
        self.iterations = 0
        t1 = time.time()
        for batch in self.train_iter:
            
            # training          
            data_X = batch.get_data()
            data_y = batch.get_label()
            t2 = time.time()
            io_time = t2 - t1
            training_loss, training_accuracy = training_fn(data_X, data_y)
            t1 = time.time()
            training_time = t1 - t2
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            
            epoch = self.iterations // self.train_iter.batch_per_epoch
            iteration = self.iterations % self.train_iter.batch_per_epoch
            logging.info("Total Iteration %d (Epoch %d, Iteration %d): loss = %f, accuracy = %f, training time = %f seconds, io time = %f seconds"% (self.iterations, epoch, iteration, training_loss, training_accuracy, training_time, io_time))
            
            # validation
            if self.val_iter != None and (self.iterations + 1) % self.val_interval == 0:
                val_loss = []
                val_accuracy = []
                logging.info( 'validating model...')
                val_iter.reset()
                val_iter.auto_stop = True
                val_iter.batch_per_epoch = 3 
                for batch in val_iter:
                    data_X = batch.data
                    data_y = batch.label
                    val_loss_batch, val_accuracy_batch = val_fn(data_X, data_y)
                    val_loss.append(val_loss_batch)
                    val_accuracy.append(val_accuracy_batch)
                    logging.info('Validating val batch %d'%(val_iter.batch_count))
                val_losses.append(np.mean(val_loss))
                val_accuracies.append(np.mean(val_accuracy))
                logging.info("Total Iteration %d (Epoch %d, Iteration %d), validation result: loss = %f, accuracy = %f"%(self.iterations, epoch, iteration, val_losses[-1], val_accuracies[-1]))

            # dump data and model
            if (self.iterations + 1) % self.dump_interval == 0:
                self.training_losses = training_losses
                self.training_accuracies = training_accuracies
                self.val_losses = val_losses
                self.val_accuracies = val_accuracies  
                
                # the two iterator won't be dumped
                self.train_iter = None
                self.val_iter = None              
                dump_model = '%s_Iteration_%d.model'%(model.name,self.iterations)
                dump_solver = '%s_Iteration_%d.solver.data'%(model.name,self.iterations)
                t2 = time.time()
                logging.info( 'dumping model and solver data...')
                util.io.dump(util.io.join_path(dump_path, dump_model), model)
                util.io.dump(util.io.join_path(dump_path, dump_solver), self)
                t1 = time.time()
                logging.info( 'dumping finished, time used = %f seconds...'%(t1 - t2))
                
                # restore the two iterators
                self.train_iter = train_iter
                self.val_iter = val_iter
                        
            self.iterations += 1
            # stop training if iterations are over.
            if self.iterations >= self.total_iterations:
                training_end = time.time()
                training_time = (training_end - training_start)/60
                time_unit = 'minites'
                
                if training_time >= 60:
                    training_time = training_time / 60.0
                    time_unit = 'hours'
                
                if training_time >= 24:
                    training_time = training_time / 24.0
                    time_unit = 'days' 
                logging.info('training finished, taking %f %s'%(training_time, time_unit));                
                break
                        


class MomentumGradientDescentSolver(SimpleGradientDescentSolver):
    """ gradient descent with momentum and decay"""
    def __init__(self, 
        train_iter, 
        batch_size, 
        momentum = 0.9,
        decay = 0.0,
        epochs = None,
        total_iterations = None,
        learning_rate = 0.0001,
        val_iter = None,
        val_interval = 1000,
        dump_interval = 5000,
        dump_path = '.'
        ):

        SimpleGradientDescentSolver.__init__(self,
            train_iter = train_iter, 
            val_iter = val_iter,
            batch_size = batch_size, 
            epochs = epochs,
            total_iterations = total_iterations,
            val_interval = val_interval,
            dump_interval = dump_interval,
            dump_path = dump_path
            )     
            
        self.momentum = dtype.cast(momentum, dtype.floatX)
        self.update_value = None 
        self.decay = dtype.cast(decay, dtype.floatX)
        
    def get_updates(self, model):
        params = model.params_to_be_updated
        if self.update_value == None:
            self.update_value = [p * 0 for p in params]
        self.update_value = [self.momentum * v - self.decay * self.learning_rate * p - self.learning_rate * T.grad(model.loss, p) for (p, v) in zip(params, self.update_value)]
        updates = [(p, p + v) for (p, v) in zip(params, self.update_value)]
        return updates
        

