

from util import logging
from training.util import monitor_loss
import time
import tensorflow as tf
import numpy as np 


@tf.function
def batch_loss(model, inp, aux, targ, loss_funct, opt = None):
    loss = 0
    with tf.GradientTape() as tape:
        pred = model(inp, aux, training=True)
        loss = loss_funct(targ, pred)
    if opt is not None:
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
    return loss




def training(model, nb_epochs, step_epoch, train_set, valid_set, loss_fct, valid_loss_fct, opt, patience, min_delta):

    # Keep results for plotting
    train_loss_results = []
    valid_loss_results = []
    steps_per_epoch = step_epoch

    # early stopping 
    patience_cnt = 0 

    logging.info('Training started...')
    start = time.time()

    for epoch in range(nb_epochs):
        
        ## training
        epoch_loss_avg = tf.keras.metrics.Mean()
        for (batch, (inp_tot, targ)) in enumerate(train_set.take(steps_per_epoch)):
            # define encoder and decoder inputs
            inp, aux = inp_tot[0], inp_tot[1]
            # loss for batch
            batch_loss_results = batch_loss(model, inp, aux, targ, loss_fct, opt)
            # training progress
            epoch_loss_avg.update_state(batch_loss_results)
        # collect training loss values
        train_loss_results.append(epoch_loss_avg.result())
        
        ## validation
        epoch_valid_loss_avg = tf.keras.metrics.Mean()
        for (batch, (inp_tot, targ)) in enumerate(valid_set.take(steps_per_epoch)):
            inp, aux = inp_tot[0], inp_tot[1]
            batch_loss_results = batch_loss(model, inp, aux, targ, valid_loss_fct, None)
            epoch_valid_loss_avg.update_state(batch_loss_results)
        # collect training loss values
        valid_loss_results.append(epoch_valid_loss_avg.result())

        # early stopping
        patience_cnt = monitor_loss(epoch, valid_loss_results, min_delta, patience_cnt)
        
        if patience_cnt > patience:
            logging.info("early stopping...") 
            break
        
        if epoch % 10 == 0:
            logging.info("Epoch {}: Loss MAE: {:.5f} --- Val Loss MAE: {:.5f}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_valid_loss_avg.result()))

    logging.info('Time taken to train {} sec\n'.format(time.time() - start))
    logging.info('Training finished...')
    
    return model, train_loss_results, valid_loss_results
