#encoding = utf-8

import logging
import time

import theano
import numpy as np
import theano.tensor as T

import util
dtype = util.dtype
class Solver(object):
    def __init__(self, 
        train_iter, 
        total_iterations = None,
        epoches = None, 
        learning_rate = None,
        val_iter = None,
        val_interval = 1000,
        val_batches = 100, 
        dump_interval = 5000,
        dump_path = '.',
        supervised = True
        ):
        """
        The count of 'epoches' will start from 0 every time when training is started or resumed, and 'totoal_iterations' is recommended.
        """
        self.supervised = supervised
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.total_iterations = total_iterations
        self.epoches = epoches
        self.dump_path = dump_path;
        self.dump_interval = dump_interval
        self.val_interval = val_interval
        self.val_batches = val_batches
        self.learning_rate = learning_rate
        # the name patterns of dump files
        self.model_dump_name_pattern ='%s_Iteration_%d.model'
        self.solver_dump_name_pattern = '%s_Iteration_%d.solver.data'
        
        self.training_losses = []
        self.training_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    
    def get_updates(self, model):
        raise NotImplementedError
        
    def fit(self, model, last_stop_iteration = None):
        if last_stop_iteration != None:
            self.iterations = last_stop_iteration + 1
        else:
            self.iterations = 0
        logging.info('start from iteration %d'%(self.iterations))
        
        # in case where the solver is loaded from a dump, where the __init__() won't be executed
        self.learning_rate = dtype.cast(self.learning_rate, dtype.floatX)
        
        if self.supervised:
            inputs = [model.input, model.label]
            #outputs = [model.loss, model.accuracy, model.ce]
            outputs = [model.loss, model.accuracy]
        else:
            inputs = [model.input]
            outputs = model.loss
        
        training_start = time.time()
        train_iter = self.train_iter

        # two conditions, the reach of either one will result in conclusion.
        if self.total_iterations is None and self.epoches is None:
            raise ValueError('Either epoches or total_iterations must be provided.')
        
        if self.epoches:
            train_iter.set_maximum_epoches(self.epoches)
                
        if self.total_iterations is None:
            self.total_iterations = np.infty
        
        val_iter = self.val_iter
        logging.info( 'building theano functions...')
        logging.info('building the training function of %s...' %(model.name))
        t1 = time.time()
        training_fn = theano.function(
                    inputs = inputs, 
                    outputs = outputs,
                    updates = self.get_updates(model)
        )

        t2 = time.time()
        logging.info("building finished, using %d seconds."%(t2 - t1))        
        
        
        if val_iter != None:    
            t1 = time.time() 
            logging.info('building the val function of %s...' %(model.name))
            
            val_fn = theano.function(
                    inputs = inputs, 
                    outputs = outputs
            )
            t2 = time.time()
            logging.info("building finished, using %d seconds."%(t2 - t1))
            
            
        get_param_count_fn = model.get_param_count_fn()
        
        logging.info('All functions are built.')

        logging.info("There are %d parameters in %s"%( get_param_count_fn(), model.name))

        training_losses = []
        training_accuracies = []
        val_losses = []
        val_accuracies = []
        dump_path = self.dump_path
        
        t1 = time.time()
        for batch in self.train_iter:
            
            # training          
            data_X = batch.get_data()
            data_y = batch.get_label()
            t2 = time.time()
            io_time = t2 - t1
            if self.supervised:
                logging.debug('forwarding and then backpropogating...')
                training_loss, training_accuracy = training_fn(data_X, data_y)
                training_accuracies.append(training_accuracy)
                t1 = time.time()
                training_time = t1 - t2
                logging.info("Total Iterations:%d, epoch %d: loss = %f, accuracy = %f, learning_rate = %f, training time = %f seconds, io time = %f seconds"% (self.iterations, train_iter.get_current_epoch(), training_loss, training_accuracy, self.learning_rate, training_time, io_time))
            else: # no accuracy for unsupervised learning
                training_loss = training_fn(data_X)
                t1 = time.time()
                training_time = t1 - t2
                logging.info("Total Iterations %d: loss = %f, learning_rate = %f, training time = %f seconds, io time = %f seconds"% (self.iterations, training_loss, self.learning_rate, training_time, io_time))
            
            training_losses.append(training_loss)
                        
            # validation
            if self.val_iter != None and (self.iterations + 1) % self.val_interval == 0:
                val_loss = []
                val_accuracy = []
                logging.info( 'validating model...')
                batch_count = 0
                while batch_count < self.val_batches:
                    batch = val_iter.next()
                    data_X = batch.get_data()
                    data_y = batch.get_label()
                    if self.supervised:
                        val_loss_batch, val_accuracy_batch = val_fn(data_X, data_y)
                        val_accuracy.append(val_accuracy_batch)
                    else:
                        val_loss_batch = val_fn(data_X)
                        
                    val_loss.append(val_loss_batch)
                    logging.info('Validating val batch %d'%(batch_count))
                
                    batch_count += 1

                val_losses.append(np.mean(val_loss))
                val_accuracies.append(np.mean(val_accuracy))
                logging.info("Total Iteration %d, validation result: loss = %f, accuracy = %f"%(self.iterations, val_losses[-1], val_accuracies[-1]))

            # dump data and model
            if (self.iterations + 1) % self.dump_interval == 0:
                self.training_losses = training_losses
                self.training_accuracies = training_accuracies
                self.val_losses = val_losses
                self.val_accuracies = val_accuracies
                
                # the two iterators and the update_value won't be dumped
                self.train_iter = None
                self.val_iter = None
                update_value = self.update_value 
                self.update_value = None

                dump_model = self.model_dump_name_pattern%(model.name,self.iterations)
                dump_solver = self.solver_dump_name_pattern%(model.name,self.iterations)
                t2 = time.time()
                logging.info( 'dumping model and solver data...')
                util.io.dump(util.io.join_path(dump_path, dump_model), model)
                util.io.dump(util.io.join_path(dump_path, dump_solver), self)
                t1 = time.time()
                logging.info( 'dumping finished, time used = %f seconds...'%(t1 - t2))
                
                # restore
                self.train_iter = train_iter
                self.val_iter = val_iter
                self.update_value = update_value
                
                
            self.iterations += 1
            # stop training if iterations are over.
            if self.iterations >= self.total_iterations:
                break
                
        training_end = time.time()
        training_time = training_end - training_start
        logging.debug(training_start)
        logging.debug(training_end)
        
        training_time = training_time / 60.0
        time_unit = 'minutes'
        
        if training_time >= 60:
            training_time = training_time / 60.0
            time_unit = 'hours'
        
        if training_time >= 24:
            training_time = training_time / 24.0
            time_unit = 'days' 
        logging.info('training finished, taking %f %s'%(training_time, time_unit));                
                
                

        
class SimpleGradientDescentSolver(Solver):
    """The most simple gradient descent model, with learning rate fixed after chosen."""
    def __init__(self, 
        train_iter, 
        total_iterations = None,
        learning_rate = 0.0001,
        val_iter = None,
        epoches = None, 
        val_interval = 1000,
        dump_interval = 5000,
        dump_path = '.',
        supervised = True
        ):

        Solver.__init__(self,
            train_iter = train_iter, 
            learning_rate = learning_rate,
            epoches = epoches,
            total_iterations = total_iterations,
            val_interval = val_interval,
            dump_interval = dump_interval,
            val_iter = val_iter,
            dump_path = dump_path,
            supervised = supervised
            )       
    
    def get_updates(self, model):
        params = model.params_to_be_updated
        updates = [(p, p - self.learning_rate * T.grad(model.loss, p)) for p in params]
        return updates 

class MomentumGradientDescentSolver(SimpleGradientDescentSolver):
    """ gradient descent with momentum and decay"""
    def __init__(self, 
        train_iter, 
        momentum = 0.9,
        decay = 0.0,
        epoches = None, 
        total_iterations = None,
        learning_rate = 0.001,
        val_iter = None,
        val_interval = 1000,
        dump_interval = 5000,
        dump_path = '.',
        supervised = True
        ):

        SimpleGradientDescentSolver.__init__(self,
            train_iter = train_iter, 
            val_iter = val_iter,
            epoches = epoches, 
            learning_rate = learning_rate,
            total_iterations = total_iterations,
            val_interval = val_interval,
            dump_interval = dump_interval,
            dump_path = dump_path,
            supervised = supervised
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
        

