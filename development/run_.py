
import yaml
from util import logging
from data.data_module import data_reader, feat_engin, data_split_scale, data_loader
from model.model_module import LSTM_ED
from training.train_module import training
from training.util import opt, loss_fct, valid_loss_fct
from testing.test_module import testing
import time

with open('config.yaml') as file:
        config = yaml.safe_load(file)


path = config['script_path']
mean_value = config['data']['threshold']
n_feat_time = config['data']['time_feature']
validation_period = config['data']['validation']
testing_period = config['data']['testing']

inp_sqc = config['loader']['input_sqc']
out_sqc = config['loader']['output_sqc']
total_dim = config['loader']['tot_dim']
aux_dim = n_feat_time
batch_tr = config['loader']['batch_tr']
batch_vl = config['loader']['batch_tr']
batch_ts =  config['loader']['batch_ts']

hidd_dim = config['model']['hidd_dim']
nb_epchs = config['model']['nb_epochs']
rcr = config['model']['rcr_init']
krnl = config['model']['krnl_reg']
dr = config['model']['dropout']
patience = config['tr_phs']['patience']
delta = config['tr_phs']['min_delta']


def main():

        # read OBU data file
        df = data_reader(path)
        # select meaninful streets (with variance) and perform feature engineering
        df_new = feat_engin(mean_value, df)
        # split and scale the data
        train, val, test, scaler = data_split_scale(df_new, validation_period, testing_period, n_feat_time)

        # transform the data to tensors
        tensor_train = data_loader(train, inp_sqc, out_sqc, aux_dim, batch_tr)
        tensor_valid = data_loader(val, inp_sqc, out_sqc, aux_dim, batch_vl)
        tensor_test = data_loader(test, inp_sqc, out_sqc, aux_dim, batch_ts)
        logging.info("-- prepare pipeline for tf")

        # define the DL model
        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)

        # start training
        step_epoch = len(train) // batch_tr
        lstm_ed, tr_loss_res, val_loss_res = training(lstm_ed, nb_epchs, step_epoch, # model, number of epochs, steps per epoch
                tensor_train,  tensor_valid, # training and validation tensors
                loss_fct, valid_loss_fct, opt, # loss functions and optimizer
                patience, delta) # early stopping
        # start testing
        pred, targ, rmse, mae = testing(lstm_ed, tensor_test, aux_dim, scaler, out_sqc)

        logging.info("Finally, I can eat my pizza(s)")

if __name__ == "__main__":
        main()