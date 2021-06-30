from util import logging
import tensorflow as tf
from testing.util import inverse_transform, evaluation_fct
import numpy as np

def testing(model, tensor_test, aux_dim, scaler, out_sqc):

    logging.info('Testing started')
    forecasts = []
    targets = []
    rmse_list = []
    mae_list = []

    for (step, (inp_tot, targ)) in enumerate(tensor_test):

            inp, aux = inp_tot[0], inp_tot[1]
            targ = tf.cast(targ, tf.float32)
            pred = model(inp, aux, training=False)
            
            truth = inverse_transform(targ[0][:,:- aux_dim],  scaler)
            pred = inverse_transform(pred[0][:,:-aux_dim],  scaler)
            
            forecasts.append(pred)
            targets.append(truth)
            
            rmse, mae = evaluation_fct(targets, forecasts, out_sqc)
            logging.info(' -- step '+ str(step)+' mae: ' +str(np.mean(mae))+' rmse: '+str(np.mean(rmse)))
            
            rmse_list.append(rmse)
            mae_list.append(mae)

    logging.info('Testing finished') 

    return forecasts, targets, rmse_list, mae_list