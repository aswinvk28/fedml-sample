import logging

import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    
class MyModelTrainer(ModelTrainer):
  
  def get_model_params(self):
    return self.model.network.state_dict()

  def set_model_params(self, model_parameters):
    self.model.network.load_state_dict(model_parameters)

  def train(self, train_data, device, args, client=None):
    model = self.model

    model.network.to(device)
    # model.train()
    
    X_train, y_train = train_data
    X_test, y_test = client.local_test_data
    
    # fit the model 
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_test.values, y_test.values)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=1, patience=50,
        batch_size=11, virtual_batch_size=11,
        num_workers=10,
        weights=1,
        drop_last=False
    )
    
  def test(self, test_data, device, args):
    model = self.model

    model.network.to(device)

    metrics = {
        'test_correct': 0,
        'test_loss': 0,
        'test_precision': 0,
        'test_recall': 0,
        'test_total': 0
    }

    X_test, y_test = test_data
    preds = model.predict(X_test.values)
    metrics['test_correct'] = np.round(accuracy_score(y_test.values, preds) * len(X_test))
    metrics['test_total'] = len(X_test)
    metrics['test_loss'] = f1_score(y_test.values, preds, average='micro')
    
    return metrics
    
  def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
    return False
  
