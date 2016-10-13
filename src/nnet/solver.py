#encoding = utf-8
import logging
import time
class Solver(object):
    def __init__(self, 
        train_iter, 
        batch_size, 
        epochs,
        learning_rate = 0.0001,
        val_iter = None,
        dump_path = '.'
        ):
        
        self.train_iter = train_iter
        self.batch_size = batch_size
        self.epochs = epochs        
        self.learning_rate = learning_rate
        self.val_iter = val_iter
        self.dump_path = dump_path;
        
    def fit(self, model):
        pass
        
class SGDSolver(Solver):
    def __init__(self, 
        train_iter, 
        batch_size, 
        epochs,
        learning_rate = 0.0001,
        val_iter = None,
        dump_path = '.'
        ):

        Solver.__init__(self,
            train_iter = train_iter, 
            batch_size = batch_size, 
            epochs = epochs,
            val_iter = val_iter,
            dump_path = dump_path
            )       

    def fit(self, model):
        logging.info( 'building theano functions...')
        
        t1 = time.time()

        training_fn = model.get_training_fn()
        val_fn = model.get_val_fn()
        get_param_count_fn = model.get_param_count_fn()
        
        t2 = time.time()
        logging.info('All functions are built, using %f seconds' % (t2 - t1))

        logging.info("There are %d parameters in %s"%( get_param_count_fn(), model.name))


        training_losses = []
        training_accuracies = []
        val_losses = []
        val_accuracies = []
        epoch_time = []
        dump_path = self.dump_path
        batch_size = self.batch_size

        for epoch in xrange(self.epochs):
            iteration = 0
            self.train_iter.reset()
            t1 = time.time()
            for batch in self.train_iter:
                data_X = batch.get_data()
                data_y = batch.get_label()
                t2 = time.time()
                io_time = t2 - t1
                training_loss, training_accuracy = training_fn(data_X, data_y, self.learning_rate)
                t1 = time.time()
                training_time = t1 - t2
                training_losses.append(training_loss)
                training_accuracies.append(training_accuracy)
                
                logging.info("Epoch %d, Iteration %d: loss = %f, accuracy = %f, training time = %f seconds, io time = %f seconds" % (epoch, iteration, training_loss, training_accuracy, training_time, io_time))
                iteration += 1
            
            # validation
            val_loss = []
            val_accuracy = []
            if val_iter != None:
                logging.info( 'validating model...')
                for batch in val_iter:
                    data_X = batch.data
                    data_y = batch.label
                    val_loss_batch, val_accuracy_batch = train_alexnet(data_X, data_y, self.learning_rate)
                    val_loss.append(val_loss_batch)
                    val_accuracy.append(val_accuracy_batch)
                val_iter.reset()
                val_losses.append(np.mean(val_loss))
                val_accuracies.append(np.mean(val_accuracy))
                logging.info( "Epoch %d, validation: loss = %f, accuracy = %f"%(epoch, val_losses[-1], val_accuracies[-1]))
            
            # dump data and model
            dump_model = '%s_Epoch%d.model'%(self.name,self.prefix,epoch)
            dump_data = '%s_Epoch%d.data'%(self.name,epoch)
            t2 = time.time()
            logging.info( 'dumping model and data...')
            util.io.dump(util.io.join_path(dump_path, dump_model), net)
            util.io.dump(util.io.join_path(dump_path, dump_data), {
                                                                    "training loss": training_losses,
                                                                    "training accuracy": training_accuracies,
                                                                    'validation loss': val_losses,
                                                                    'validation accuracies': val_accuracies
                                                                    })
            t1 = time.time()
            logging.info( 'dumping finished, time used = %f seconds...'%(t1 - t2))            
