#!/usr/bin/env python
# coding: utf-8

# In[4]:
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoModel, AutoTokenizer, AdamW
from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import  load_from_disk
from transformers import AutoConfig, DataCollatorForTokenClassification
from transformers import get_cosine_schedule_with_warmup


# In[9]:


class CFG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 69
    # dataset path 
    train_dataset_path = "/workspace/train.json"
    external_dataset_path_1 = "/workspace/mpware_mixtral8x7b_v1.1-no-i-username.json"
    save_dir="/workspace/exp1"

    #tokenizer params
    downsample = 0.45
    truncation = True 
    padding = False #'max_length'
    max_length = 1024
    freeze_layers = 0
    # model params
    model_name = "Qwen/Qwen1.5-0.5B"
    
    target_cols = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL','O']

    load_from_disk = None
    #training params
    learning_rate = 1e-5
    batch_size = 4
    epochs = 3


seed_everything(CFG.seed)

if not os.path.exists(CFG.save_dir):
  os.makedirs(CFG.save_dir)


# In[15]:


all_labels = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

ds = load_from_disk("/workspace/pii-ds")

class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out

    
class PIIModel(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.cfg = config
        self.model_config = AutoConfig.from_pretrained(
            config.model_name,
        )

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        self.model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )

        self.transformers_model = AutoModel.from_pretrained(self.cfg.model_name,config=self.model_config)
        self.head = LSTMHead(in_features=self.model_config.hidden_size, hidden_dim=self.model_config.hidden_size//2, n_layers=1)

        self.output = nn.Linear(self.model_config.hidden_size, len(self.cfg.target_cols))
        
        if self.cfg.freeze_layers>0:
            print(f'Freezing {self.cfg.freeze_layers} layers.')
            for layer in self.transformers_model.longformer.encoder.layer[:self.cfg.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False


        self.loss_function = nn.CrossEntropyLoss(reduction='mean',ignore_index=-100) 
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask,train):
        
        transformer_out = self.transformers_model(input_ids,attention_mask = attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.head(sequence_output)
        logits = self.output(sequence_output)

        return (logits, None)
    

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['labels'] 

        outputs = self(input_ids,attention_mask,train=True)
        output = outputs[0]
        loss = self.loss_function(output.view(-1,len(self.cfg.target_cols)), target.view(-1))
        
        self.log('train_loss', loss , prog_bar=True)
        return {'loss': loss}
    
    def train_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f'epoch {trainer.current_epoch} training loss {avg_loss}')
        return {'train_loss': avg_loss} 
        
    def train_dataloader(self):
        return self._train_dataloader 

    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in self.transformers_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.transformers_model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters() if "transformers_model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.cfg.learning_rate)

        epoch_steps = self.cfg.data_length
        batch_size = self.cfg.batch_size

        warmup_steps = 0.05 * epoch_steps // batch_size
        training_steps = self.cfg.epochs * epoch_steps // batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, num_cycles=0.5)
        
        lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


# In[22]:


collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=512)


# In[23]:


keep_cols = {"input_ids", "attention_mask", "labels"}
train_ds = ds
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
CFG.data_length = len(train_ds)
CFG.len_token = len(tokenizer)


# In[24]:


print('Dataset Loaded....')
print((train_ds[0].keys()))
print("Generating Train DataLoader")
train_dataloader = DataLoader(train_ds, batch_size = CFG.batch_size, shuffle = True, num_workers= 4, pin_memory=False,collate_fn = collator)


# In[25]:


early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=8, verbose= True, mode="min")
checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          dirpath= CFG.save_dir,
                                      save_top_k=1,
                                      save_last= True,
                                      save_weights_only=True,
                                      verbose= True,
                                      mode='min')

wandb_logger = WandbLogger()
    
print("Model Creation")


# In[20]:

torch.set_float32_matmul_precision('medium')
model = PIIModel(CFG)
# model.load_state_dict(torch.load('/home/nischay/PID/nbs/outputs2/exp12_baseline_debv3base_1024_extv1/ckeckpoint_0-v2.ckpt','cpu')['state_dict'])
trainer = Trainer(max_epochs= CFG.epochs,
                      deterministic=False,
                      accumulate_grad_batches=1, 
                      devices=4,
                      precision='bf16-mixed', 
                      strategy='ddp',
                      logger=wandb_logger,
                      accelerator=CFG.device ,
                      callbacks=[checkpoint_callback,early_stop_callback]) 


# In[21]:


CFG.data_length = len(train_ds)
trainer.fit(model,train_dataloader)
