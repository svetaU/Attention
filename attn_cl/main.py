import numpy as np
import pandas as pd
import pickle
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data as data
import torch.optim as optim
from torchinfo import summary
import os
from datasets import FakeDataset
from plotting_functions import *
import model_classes as mc
CHECKPOINT_PATH = "/attn_classifier/chk_pt_1/"
os.environ["TORCH_HOME"] = CHECKPOINT_PATH
device = torch.device("cpu")

grad_file_name = "grads.pkl"
loss_file_name = "loss.pkl"
scores_file_name = "scores.pkl"
maps_file_name = "maps.pkl"
data_file_name = "input_data.png"
OUTPUT_EVERY = 1
NUM_ITERATIONS = 10

def cycle(loader):
    while True:
        for data in loader:
            yield data

train_ds = FakeDataset(600,9,64,0.5)
train_ds_loader = cycle(data.DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=False))
val_ds = FakeDataset(100,9,64,0.3)
val_ds_loader = cycle(data.DataLoader(val_ds, batch_size=16, shuffle=False,  drop_last=False))
test_ds = FakeDataset(300,9,64,0.5)
test_ds_loader = cycle(data.DataLoader(test_ds, batch_size=16, shuffle=False,  drop_last=False))

def manual_train():
    num_batches = NUM_ITERATIONS
    out_dir = os.path.join(CHECKPOINT_PATH, "train_output")
    os.makedirs(out_dir, exist_ok=True)

    classifier = mc.AttnClassifier(embed_dim=64,
                                   model_dim=128,
                                   num_tokens = 9,
                                   num_heads=4,
                                   num_layers=4,
                                   dropout=0.1,
                                   input_dropout=0.1)
    model = mc.TrainerWrapper(classifier)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    #model.to("mps")
    model.train()
    training_loss = []
    scores = []
    param_grads = {}
    maps = {}
    data_iter = iter(train_ds_loader)
    test_iter = iter(test_ds_loader)
    #Visualize data
    batch = next(data_iter)
    plot_input_data(batch, out_dir + "/" + data_file_name)
    #summary(model)
    for i in range(num_batches):
        loss, ac, prd, s1, s2 = model(next(data_iter))
        loss.backward()
        if i%OUTPUT_EVERY == 0:
            training_loss.append((i, loss.detach().numpy().item()))
            scores.append((i,s1.detach().cpu().numpy(),s2.detach().cpu().numpy()))
            model.eval()
            test_data, test_labels = next(test_iter)
            attn_map = model.get_attention_maps(test_data, add_positional_encoding=False)
            maps[i] = (attn_map, test_labels)
            model.train()
            for name, p in model.named_parameters():
                if name not in param_grads.keys():
                    param_grads[name] = []
                param_grads[name].append((p.grad.detach().cpu().numpy().max(),
                                          p.grad.detach().numpy().min(),
                                          p.grad.detach().numpy().mean()))

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    #model = model.to(device)
    with open(out_dir + "/" + grad_file_name, 'wb') as file:
        pickle.dump(param_grads, file)
    with open(out_dir + "/" + loss_file_name, 'wb') as file:
        pickle.dump(training_loss, file)
    with open(out_dir + "/" + scores_file_name, 'wb') as file:
        pickle.dump(scores, file)
    with open(out_dir + "/" + maps_file_name, 'wb') as file:
        pickle.dump(maps, file)

    #plot_param_grads(param_grads, "net.input_net.1.weight")
    #plot_loss(training_loss)
    #plot_scores(scores)
    #inp_data, test_labels = next(test_iter)
    #attention_maps = model.get_attention_maps(inp_data, add_positional_encoding=False)
    #plot_attention_maps(input_data=None,attn_maps=attention_maps,idx=15)
    #plot_input_data(next(test_iter))
    #for i in range(0,6):
    #    acc, preds, labels = model.acc(next(test_iter))
    #    print(labels.detach().cpu().numpy())
    #    print(preds.view(-1).detach().cpu().numpy())
    #    print(f'Test accuracy {acc.detach().cpu().numpy()*100}%')

    return 0


if __name__ == "__main__":
    manual_train()
