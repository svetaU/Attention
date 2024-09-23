import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
from graphviz import Digraph
import imageio
import os
#plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf') # For export
#import matplotlib_inline
#matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
#import matplotlib
#matplotlib.rcParams['lines.linewidth'] = 2.0
#import seaborn as sns
#sns.reset_orig()

GRADIENT_SCALE = 50
GRADIENT_SCALE2 = 10

def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def plot_attention_maps_file(filename, iteration = 0, idx = 0):

    image_file_name = filename.replace('.pkl',f'_{iteration}_{idx}.png')

    with open(filename, 'rb') as file:
        attn_data = pickle.load(file)
    attn_maps, labels = attn_data[iteration]
    input_data = np.arange(attn_maps[0][idx].shape[-1])
    label = int(labels[idx])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    fig.suptitle(f'Class label {label}')
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig(image_file_name)


def plot_param_grads(data, param_name):
    param_df = pd.DataFrame(data[param_name], columns=['param_max', 'param_min', 'param_mean'])
    ax = param_df.plot(title=param_name, use_index=True)
    plt.show()
    return 0

def plot_loss(data):
    param_df = pd.DataFrame(data, columns=['iter_step', 'loss'])
    ax = param_df.plot(x='iter_step', y='loss', title="Training Loss", use_index=False)
    plt.show()
    return ax

def plot_loss_file(filename):
    image_file_name = filename.replace('pkl','png')
    with open(filename, 'rb') as file:
        loss_data = pickle.load(file)
    param_df = pd.DataFrame(loss_data, columns=['iter_step', 'loss'])
    ax = param_df.plot(x='iter_step', y='loss', title="Training Loss")
    ax.set_xlabel("Step")
    plt.show()
    fig = ax.get_figure()
    fig.savefig(image_file_name)
    return 0


def plot_param_grads_file(filename, param_name):
    with open(filename, 'rb') as file:
        param_dict = pickle.load(file)
    if param_name not in param_dict.keys():
        print(f'{param_name} not found in data!')
        return -1
    image_file_name = filename.replace('.pkl',f'{param_name}.png')
    param_df = pd.DataFrame(param_dict[param_name], columns=['param_max', 'param_min', 'param_mean'])
    #ax = param_df.plot(title=param_name, use_index=True)
    fig, ax = plt.subplots(1,1)
    ax.plot(param_df["param_max"], color='orange', label = 'Max')
    ax.plot(param_df["param_min"], color='blue', label = 'Min')
    ax.plot(param_df["param_mean"], color='green', label = 'Avg')
    ax.set_title(f'Gradients for {param_name}')
    ax.set_xlabel("Step")
    ax.legend()
    left, bottom, width, height = [0.65, 0.2, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(param_df["param_max"][:210], color='orange')
    ax2.plot(param_df["param_min"][:210], color='blue')
    ax2.plot(param_df["param_mean"][:210], color='green')
    plt.show()
    fig.savefig(image_file_name)
    return 0

def plot_scores(data):
    score_data = pd.DataFrame(data, columns=['iter_step', 'max_score', 'min_score'])
    fig, ax = plt.subplots(2,1)
    ax[0].plot(score_data["iter_step"], score_data["max_score"], color='orange')
    ax[0].plot(score_data["iter_step"], score_data["min_score"], color='blue')
    ax[1].plot(score_data['max_score'], score_data['max_score'] - score_data['min_score'])
    plt.show()
    return ax

def plot_scores_file(filename):
    image_file_name = filename.replace('.pkl','.png')
    with open(filename, 'rb') as file:
        scores = pickle.load(file)
    score_data = pd.DataFrame(scores, columns=['iter_step', 'max_score', 'min_score'])
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1, hspace = 0.5, wspace = 0)
    ax = gs.subplots()
    ax.plot(score_data["iter_step"], score_data["max_score"], color='orange', label = "Class1")
    ax.plot(score_data["iter_step"], score_data["min_score"], color='blue', label = "Class2")
    ax.set_title("Class scores (mean over batch)")
    ax.set_xlabel("Step")
    ax.legend()
    left, bottom, width, height = [0.65, 0.6, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(score_data["iter_step"][:100], score_data["max_score"][:100], color='orange')
    ax2.plot(score_data["iter_step"][:100], score_data["min_score"][:100], color='blue')
    #ax[1].plot(score_data['max_score'], score_data['max_score'] - score_data['min_score'])
    #ax[1].set_title("Scores difference")
    #ax[1].set_xlabel("Class1 mean score")
    plt.show()
    fig.savefig(image_file_name)

def plot_transformer_layer(sub_g, layer_data, layer_name, iteration):

    gradient_strength = ['blue','slateblue','mediumslateblue','mediumpurple','purple','red3','red2','red1','red']

    suffx = f'{layer_name}_{iteration}'

    params_to_nodes = {
        'attn_layer.to_qvk.weight':'QVK_weights' + suffx,
        'attn_layer.last_linear.weight':'Feed_FWD' + suffx,
        'norm1.weight':'Norm1_weight' + suffx,
        'norm1.bias':'Norm1_bias' + suffx,
        'norm2.weight':'Norm2_weight' + suffx,
        'norm2.bias':'Norm2_bias' + suffx,
        'linear_net.0.weight':'FeedFwd2_weight' + suffx,
        'linear_net.0.bias':'FeedFwd2_bias' + suffx,
        'linear_net.3.weight':'FeedFwd3_weight' + suffx,
        'linear_net.3.bias':'FeedFwd3_bias' + suffx
    }

    node_colors = {}
    for key in layer_data.keys():
        param_name = key.replace(layer_name,'')
        #print(f'param name \"{param_name}\", node name \"{params_to_nodes[param_name]}\"')
        p_df = pd.DataFrame(layer_data[key], columns=['param_max', 'param_min', 'param_mean'])
        max_val = p_df.iloc[[iteration]]['param_max'].squeeze()
        min_val = p_df.iloc[[iteration]]['param_min'].squeeze()
        if min_val == max_val:
            diff = abs(max_val)
        else:
            diff = abs(max_val - min_val)
        #diff = abs(p_df.iloc[[iteration]]['param_mean'].squeeze())
        color_index = max(math.floor((1./(1. + math.exp(-diff*GRADIENT_SCALE)) - 0.5)*18),0)
        node_colors[params_to_nodes[param_name]] = min(color_index, len(gradient_strength) - 1)
        #print(color_index)

    sub_g.node('Input' + suffx, 'Input')

    with sub_g.subgraph(name = 'cluster_attn') as c:
        c.attr(label = 'Attention')
        #Trained nodes
        c.node('QVK_weights' + suffx, 'QVK_weights', fillcolor = f"{gradient_strength[node_colors['QVK_weights' + suffx]]}")
        c.node('Feed_FWD' + suffx, 'FeedFwr_W', fillcolor = f"{gradient_strength[node_colors['Feed_FWD' + suffx]]}")

    #Non-trainables
    sub_g.node('Drop1' + suffx,'Drop')
    sub_g.node('Drop2' + suffx,'Drop')
    sub_g.node('ReLu1' + suffx,'ReLu')
    sub_g.node('Add1' + suffx,'Add')
    sub_g.node('Add2' + suffx,'Add')
    sub_g.node('Add3' + suffx,'Add')
    sub_g.node('Add4' + suffx,'Add')

    #Trained nodes
    sub_g.node('Norm1_weight' + suffx,'LayerNorm_W', fillcolor = f"{gradient_strength[node_colors['Norm1_weight' + suffx]]}")
    sub_g.node('Norm1_bias' + suffx, 'LayerNorm_B', fillcolor = f"{gradient_strength[node_colors['Norm1_bias' + suffx]]}")
    sub_g.node('Norm2_weight' + suffx,'LayerNorm_W', fillcolor = f"{gradient_strength[node_colors['Norm2_weight' + suffx]]}")
    sub_g.node('Norm2_bias' + suffx, 'LayerNorm_B', fillcolor = f"{gradient_strength[node_colors['Norm2_bias' + suffx]]}")
    sub_g.node('FeedFwd2_weight' + suffx, 'FeedFwr_W', fillcolor = f"{gradient_strength[node_colors['FeedFwd2_weight' + suffx]]}")
    sub_g.node('FeedFwd2_bias' + suffx, 'FeedFwr_B', fillcolor = f"{gradient_strength[node_colors['FeedFwd2_bias' + suffx]]}")
    sub_g.node('FeedFwd3_weight' + suffx, 'FeedFwr_W', fillcolor = f"{gradient_strength[node_colors['FeedFwd3_weight' + suffx]]}")
    sub_g.node('FeedFwd3_bias' + suffx, 'FeedFwr_B', fillcolor = f"{gradient_strength[node_colors['FeedFwd3_bias' + suffx]]}")

    #Transformer edges
    sub_g.edge('Input' + suffx,'QVK_weights' + suffx)
    sub_g.edge('QVK_weights' + suffx,'Feed_FWD' + suffx)
    sub_g.edge('Input' + suffx,'Drop1' + suffx)
    sub_g.edge('Feed_FWD' + suffx,'Drop1' + suffx)
    sub_g.edge('Drop1' + suffx,'Norm1_weight' + suffx)
    sub_g.edge('Drop1' + suffx,'Norm1_bias' + suffx)
    sub_g.edge('Norm1_weight' + suffx,'Add1' + suffx)
    sub_g.edge('Norm1_bias' + suffx,'Add1' + suffx)
    sub_g.edge('Add1' + suffx,'FeedFwd2_weight' + suffx)
    sub_g.edge('Add1' + suffx,'FeedFwd2_bias' + suffx)
    sub_g.edge('FeedFwd2_weight' + suffx,'Add2' + suffx)
    sub_g.edge('FeedFwd2_bias' + suffx,'Add2' + suffx)
    sub_g.edge('Add2' + suffx,'Drop2' + suffx)
    sub_g.edge('Drop2' + suffx,'ReLu1' + suffx)
    sub_g.edge('ReLu1' + suffx,'FeedFwd3_weight' + suffx)
    sub_g.edge('ReLu1' + suffx,'FeedFwd3_bias' + suffx)
    sub_g.edge('FeedFwd3_weight' + suffx, 'Add3' + suffx)
    sub_g.edge('FeedFwd3_bias' + suffx, 'Add3' + suffx)
    sub_g.edge('Add3' + suffx,'Norm2_weight' + suffx)
    sub_g.edge('Add3' + suffx,'Norm2_bias' + suffx)
    sub_g.edge('Norm2_weight' + suffx,'Add4' + suffx)
    sub_g.edge('Norm2_bias' + suffx,'Add4' + suffx)
    sub_g.edge('Add1' + suffx,'Add3' + suffx)

    return sub_g

def plot_iteration_graph_from_file(filename, iteration, output_path):

    graph_name = f'Transformer{iteration}'
    frmt = "png"
    out_filename = output_path + graph_name + "." + frmt

    with open(filename, 'rb') as file:
        param_dict = pickle.load(file)


    gradient_strength = ['blue','slateblue','mediumslateblue','mediumpurple','purple','red3','red2','red1','red']

    in_out_params_to_nodes = {
        'net.input_net.1.weight':'FeedFwd_in_weight',
        'net.input_net.1.bias':'FeedFwd_in_bias',
        'net.output_net.0.weight':'FeedFwd_out_weight1',
        'net.output_net.0.bias':'FeedFwd_out_bias1',
        'net.output_net.1.weight':'Norm_out_weight',
        'net.output_net.1.bias':'Norm_out_bias',
        'net.output_net.3.weight':'FeedFwd_out_weight2',
        'net.output_net.3.bias':'FeedFwd_out_bias2',
        'net.prob.0.weight':'FeedFwd_prob_weight1',
        'net.prob.0.bias':'FeedFwd_prob_bias1'
    }

    in_out_nodes_color = {}
    for param_name in in_out_params_to_nodes.keys():
        p_df = pd.DataFrame(param_dict[param_name], columns=['param_max', 'param_min', 'param_mean'])
        max_val = p_df.iloc[[iteration]]['param_max'].squeeze()
        min_val = p_df.iloc[[iteration]]['param_min'].squeeze()
        if min_val == max_val:
            diff = abs(max_val)
        else:
            diff = abs(max_val - min_val)
        #diff = abs(p_df.iloc[[iteration]]['param_mean'].squeeze())
        color_index = max(math.floor((1./(1. + math.exp(-diff*GRADIENT_SCALE2)) - 0.5)*18),0)
        in_out_nodes_color[in_out_params_to_nodes[param_name]] = min(color_index, len(gradient_strength) - 1)

    g = Digraph(graph_name, f'Iteration{iteration:03d}', format=frmt)
    node_attr = {'shape':'box','style':'filled'}
    g_attr = {'size':'12,24', 'label': f'Iteration{iteration:03d}','fontsize':'32'}
    g.node_attr.update(node_attr)
    g.graph_attr.update(g_attr)

    #Label
    #g.node(f'Iteration{iteration:03d}', shape="plaintext")

    #Input
    with g.subgraph(name = 'cluster_input') as c:
        c.attr(label='Input Layer')
        c.node('Drop_in', 'Drop')
        c.node('FeedFwd_in_weight', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_in_weight']]}")
        c.node('FeedFwd_in_bias', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_in_bias']]}")
        c.node('Add_in','Add')
    g.edge('Drop_in','FeedFwd_in_weight')
    g.edge('Drop_in','FeedFwd_in_bias')
    g.edge('FeedFwd_in_weight','Add_in')
    g.edge('FeedFwd_in_bias','Add_in')


    #Transformer nodes
    for layer in range(0,4):
        layer_name = f'net.transformer.layers.{layer}.'
        layer_data = dict(filter(lambda item: item[0].find(layer_name) > -1, param_dict.items()))
        with g.subgraph(name = f'cluster_{layer}') as c:
            c.attr(label = layer_name)
            plot_transformer_layer(c, layer_data, layer_name, iteration)

    #Output
    with g.subgraph(name = 'cluster_logit') as c:
        c.attr(label = 'Logit Layer')
        c.node('FeedFwd_out_weight1', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_weight1']]}")
        c.node('FeedFwd_out_bias1', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_bias1']]}")
        c.node('Add_out1','Add')
        c.node('Norm_out_weight','LayerNorm_W', fillcolor = f"{gradient_strength[in_out_nodes_color['Norm_out_weight']]}")
        c.node('Norm_out_bias', 'LayerNorm_B', fillcolor = f"{gradient_strength[in_out_nodes_color['Norm_out_bias']]}")
        c.node('Add_out2','Add')
        c.node('FeedFwd_out_weight2', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_weight2']]}")
        c.node('FeedFwd_out_bias2', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_bias2']]}")
        c.node('Add_out3','Add')
        c.node('ReLu_out1','ReLu')
        c.node('ReLu_out2','ReLu')
    g.edge('FeedFwd_out_weight1','Add_out1')
    g.edge('FeedFwd_out_bias1','Add_out1')
    g.edge('Add_out1','Norm_out_weight')
    g.edge('Add_out1','Norm_out_bias')
    g.edge('Norm_out_weight','Add_out2')
    g.edge('Norm_out_bias','Add_out2')
    g.edge('Add_out2','ReLu_out1')
    g.edge('ReLu_out1','FeedFwd_out_weight2')
    g.edge('ReLu_out1','FeedFwd_out_bias2')
    g.edge('FeedFwd_out_weight2','Add_out3')
    g.edge('FeedFwd_out_bias2','Add_out3')
    g.edge('Add_out3','ReLu_out2')

    #Probability
    with g.subgraph(name = 'cluster_prob') as c:
        c.attr(label = 'Probability Layer')
        c.node('FeedFwd_prob_weight1', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_prob_weight1']]}")
        c.node('FeedFwd_prob_bias1', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_prob_bias1']]}")
        c.node('Add_prob','Add')
        c.node('Sigmoid','Sigmoid')
    g.edge('FeedFwd_prob_weight1','Add_prob')
    g.edge('FeedFwd_prob_bias1','Add_prob')
    g.edge('Add_prob','Sigmoid')

    #Legend
    with g.subgraph(name = 'cluster_legend') as c:
        c.attr(peripheries='0', label = 'Gradient strength legend', fontsize='24')
        c.node('block1', label='', xlabel='0.00-0.11', fontsize='20', fillcolor = f'{gradient_strength[0]}')
        c.node('block2', label='', xlabel='0.11-0.22', fontsize='20', fillcolor = f'{gradient_strength[1]}')
        c.node('block3', label='', xlabel='0.22-0.33', fontsize='20', fillcolor = f'{gradient_strength[2]}')
        c.node('block4', label='', xlabel='0.33-0.44', fontsize='20', fillcolor = f'{gradient_strength[3]}')
        c.node('block5', label='', xlabel='0.44-0.55', fontsize='20', fillcolor = f'{gradient_strength[4]}')
        c.node('block6', label='', xlabel='0.55-0.66', fontsize='20', fillcolor = f'{gradient_strength[5]}')
        c.node('block7', label='', xlabel='0.66-0.77', fontsize='20', fillcolor = f'{gradient_strength[6]}')
        c.node('block8', label='', xlabel='0.77-0.88', fontsize='20', fillcolor = f'{gradient_strength[7]}')
        c.node('block9', label='', xlabel='0.88-1.00', fontsize='20', fillcolor = f'{gradient_strength[8]}')
    g.edge('Sigmoid','block1', style='invis')
    g.edge('block1','block2', style='invis')
    g.edge('block2','block3', style='invis')
    g.edge('block3','block4', style='invis')
    g.edge('block4','block5', style='invis')
    g.edge('block5','block6', style='invis')
    g.edge('block6','block7', style='invis')
    g.edge('block7','block8', style='invis')
    g.edge('block8','block9', style='invis')

    g.view()
    g.render(outfile = out_filename)
    return g

def plot_iteration_graph(param_dict, iteration, output_path):

    os.makedirs(output_path, exist_ok=True)

    graph_name = f'Transformer{iteration}'
    frmt = "png"
    out_filename = output_path + graph_name + "." + frmt

    gradient_strength = ['blue','slateblue','mediumslateblue','mediumpurple','purple','red3','red2','red1','red']

    in_out_params_to_nodes = {
        'net.input_net.1.weight':'FeedFwd_in_weight',
        'net.input_net.1.bias':'FeedFwd_in_bias',
        'net.output_net.0.weight':'FeedFwd_out_weight1',
        'net.output_net.0.bias':'FeedFwd_out_bias1',
        'net.output_net.1.weight':'Norm_out_weight',
        'net.output_net.1.bias':'Norm_out_bias',
        'net.output_net.3.weight':'FeedFwd_out_weight2',
        'net.output_net.3.bias':'FeedFwd_out_bias2',
        'net.prob.0.weight':'FeedFwd_prob_weight1',
        'net.prob.0.bias':'FeedFwd_prob_bias1'
    }

    in_out_nodes_color = {}
    for param_name in in_out_params_to_nodes.keys():
        p_df = pd.DataFrame(param_dict[param_name], columns=['param_max', 'param_min', 'param_mean'])
        max_val = p_df.iloc[[iteration]]['param_max'].squeeze()
        min_val = p_df.iloc[[iteration]]['param_min'].squeeze()
        if min_val == max_val:
            diff = abs(max_val)
        else:
            diff = abs(max_val - min_val)
        #diff = abs(p_df.iloc[[iteration]]['param_mean'].squeeze())
        color_index = max(math.floor((1./(1. + math.exp(-diff*GRADIENT_SCALE)) - 0.5)*18),0)
        in_out_nodes_color[in_out_params_to_nodes[param_name]] = min(color_index, len(gradient_strength) - 1)

    g = Digraph(graph_name, format=frmt)
    node_attr = {'shape':'box','style':'filled'}
    g_attr = {'size':'12,24', 'label': f'Iteration{iteration:03d}','fontsize':'32'}
    g.node_attr.update(node_attr)
    g.graph_attr.update(g_attr)

    #Input
    with g.subgraph(name = 'cluster_input') as c:
        c.attr(label='Input Layer')
        c.node('Drop_in', 'Drop')
        c.node('FeedFwd_in_weight', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_in_weight']]}")
        c.node('FeedFwd_in_bias', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_in_bias']]}")
        c.node('Add_in','Add')
    g.edge('Drop_in','FeedFwd_in_weight')
    g.edge('Drop_in','FeedFwd_in_bias')
    g.edge('FeedFwd_in_weight','Add_in')
    g.edge('FeedFwd_in_bias','Add_in')


    #Transformer nodes
    for layer in range(0,4):
        layer_name = f'net.transformer.layers.{layer}.'
        layer_data = dict(filter(lambda item: item[0].find(layer_name) > -1, param_dict.items()))
        with g.subgraph(name = f'cluster_{layer}') as c:
            c.attr(label = layer_name)
            plot_transformer_layer(c, layer_data, layer_name, iteration)

    #Output
    with g.subgraph(name = 'cluster_logit') as c:
        c.attr(label='Logit Layer')
        c.node('FeedFwd_out_weight1', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_weight1']]}")
        c.node('FeedFwd_out_bias1', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_bias1']]}")
        c.node('Add_out1','Add')
        c.node('Norm_out_weight','LayerNorm_W', fillcolor = f"{gradient_strength[in_out_nodes_color['Norm_out_weight']]}")
        c.node('Norm_out_bias', 'LayerNorm_B', fillcolor = f"{gradient_strength[in_out_nodes_color['Norm_out_bias']]}")
        c.node('Add_out2','Add')
        c.node('FeedFwd_out_weight2', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_weight2']]}")
        c.node('FeedFwd_out_bias2', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_out_bias2']]}")
        c.node('Add_out3','Add')
        c.node('ReLu_out1','ReLu')
        c.node('ReLu_out2','ReLu')
    g.edge('FeedFwd_out_weight1','Add_out1')
    g.edge('FeedFwd_out_bias1','Add_out1')
    g.edge('Add_out1','Norm_out_weight')
    g.edge('Add_out1','Norm_out_bias')
    g.edge('Norm_out_weight','Add_out2')
    g.edge('Norm_out_bias','Add_out2')
    g.edge('Add_out2','ReLu_out1')
    g.edge('ReLu_out1','FeedFwd_out_weight2')
    g.edge('ReLu_out1','FeedFwd_out_bias2')
    g.edge('FeedFwd_out_weight2','Add_out3')
    g.edge('FeedFwd_out_bias2','Add_out3')
    g.edge('Add_out3','ReLu_out2')

    #Probability
    with g.subgraph(name = 'cluster_prob') as c:
        c.attr(label = 'Probability Layer')
        c.node('FeedFwd_prob_weight1', 'FeedFwr_W', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_prob_weight1']]}")
        c.node('FeedFwd_prob_bias1', 'FeedFwr_B', fillcolor = f"{gradient_strength[in_out_nodes_color['FeedFwd_prob_bias1']]}")
        c.node('Add_prob','Add')
        c.node('Sigmoid','Sigmoid')
    g.edge('FeedFwd_prob_weight1','Add_prob')
    g.edge('FeedFwd_prob_bias1','Add_prob')
    g.edge('Add_prob','Sigmoid')

    #g.view()
    g.render(outfile = out_filename)
    return g

def generate_training_graph(filename, output_path):

    with open(filename, 'rb') as file:
        param_dict = pickle.load(file)

    total_iterations = 0
    for key in param_dict.keys():
        total_iterations = len(param_dict[key])
        break

    for i in range(0,total_iterations):
        plot_iteration_graph(param_dict,i,output_path)


def make_training_anim(path):

    images = []
    for i in range(0,300,1):
        file_name = path + f'Transformer{i}.png'
        images.append(imageio.v3.imread(file_name))
    imageio.mimsave(path + 'training_short.gif', images,fps=1, loop= 10)
    return 0

def plot_input_data(data, filename=None):
    inp_data, labels = data
    idx = np.where(labels == 1)[0].squeeze()[0]
    sentence = inp_data[idx].numpy().flatten()
    fig, ax = plt.subplots(2,1)
    ax[0].plot(sentence, color='orange')
    ax[0].set_title("Sequence of class 1")
    ax[0].set_ylabel("Signal")
    idx = np.where(labels == 0)[0].squeeze()[0]
    sentence1 = inp_data[idx].numpy().flatten()
    ax[1].plot(sentence1, color='orange')
    ax[1].set_title("Sequence of class 0")
    ax[1].set_ylabel("Signal")
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    if filename is not None:
        fig.savefig(filename)
    return 0

def plot_transformer_model_layer(sub_g, suffx):
    sub_g.node('Input' + suffx, 'Input')

    with sub_g.subgraph(name = 'cluster_attn') as c:
        c.attr(label = 'Attention')
        #Trained nodes
        c.node('QVK_weights' + suffx, 'QVK_weights')
        c.node('Feed_FWD' + suffx, 'FeedFwr_W')

    #Non-trainables
    sub_g.node('Drop1' + suffx,'Drop')
    sub_g.node('Drop2' + suffx,'Drop')
    sub_g.node('ReLu1' + suffx,'ReLu')
    sub_g.node('Add1' + suffx,'Add')
    sub_g.node('Add2' + suffx,'Add')
    sub_g.node('Add3' + suffx,'Add')
    sub_g.node('Add4' + suffx,'Add')

    #Trained nodes
    sub_g.node('Norm1_weight' + suffx,'LayerNorm_W')
    sub_g.node('Norm1_bias' + suffx, 'LayerNorm_B')
    sub_g.node('Norm2_weight' + suffx,'LayerNorm_W')
    sub_g.node('Norm2_bias' + suffx, 'LayerNorm_B')
    sub_g.node('FeedFwd2_weight' + suffx, 'FeedFwr_W')
    sub_g.node('FeedFwd2_bias' + suffx, 'FeedFwr_B')
    sub_g.node('FeedFwd3_weight' + suffx, 'FeedFwr_W')
    sub_g.node('FeedFwd3_bias' + suffx, 'FeedFwr_B')

    #Transformer edges
    sub_g.edge('Input' + suffx,'QVK_weights' + suffx)
    sub_g.edge('QVK_weights' + suffx,'Feed_FWD' + suffx)
    sub_g.edge('Input' + suffx,'Drop1' + suffx)
    sub_g.edge('Feed_FWD' + suffx,'Drop1' + suffx)
    sub_g.edge('Drop1' + suffx,'Norm1_weight' + suffx)
    sub_g.edge('Drop1' + suffx,'Norm1_bias' + suffx)
    sub_g.edge('Norm1_weight' + suffx,'Add1' + suffx)
    sub_g.edge('Norm1_bias' + suffx,'Add1' + suffx)
    sub_g.edge('Add1' + suffx,'FeedFwd2_weight' + suffx)
    sub_g.edge('Add1' + suffx,'FeedFwd2_bias' + suffx)
    sub_g.edge('FeedFwd2_weight' + suffx,'Add2' + suffx)
    sub_g.edge('FeedFwd2_bias' + suffx,'Add2' + suffx)
    sub_g.edge('Add2' + suffx,'Drop2' + suffx)
    sub_g.edge('Drop2' + suffx,'ReLu1' + suffx)
    sub_g.edge('ReLu1' + suffx,'FeedFwd3_weight' + suffx)
    sub_g.edge('ReLu1' + suffx,'FeedFwd3_bias' + suffx)
    sub_g.edge('FeedFwd3_weight' + suffx, 'Add3' + suffx)
    sub_g.edge('FeedFwd3_bias' + suffx, 'Add3' + suffx)
    sub_g.edge('Add3' + suffx,'Norm2_weight' + suffx)
    sub_g.edge('Add3' + suffx,'Norm2_bias' + suffx)
    sub_g.edge('Norm2_weight' + suffx,'Add4' + suffx)
    sub_g.edge('Norm2_bias' + suffx,'Add4' + suffx)
    sub_g.edge('Add1' + suffx,'Add3' + suffx)

    return sub_g

def plot_model(output_path):

    os.makedirs(output_path, exist_ok=True)

    graph_name = f'TransformerModel'
    frmt = "png"
    out_filename = output_path + graph_name + "." + frmt


    g = Digraph(graph_name, format=frmt)
    node_attr = {'shape':'box','style':'filled'}
    g_attr = {'size':'12,24', 'label': f'TransformerModel','fontsize':'32'}
    g.node_attr.update(node_attr)
    g.graph_attr.update(g_attr)

    #Input
    with g.subgraph(name = 'cluster_input') as c:
        c.attr(label='Input Layer')
        c.node('Drop_in', 'Drop')
        c.node('FeedFwd_in_weight', 'FeedFwr_W')
        c.node('FeedFwd_in_bias', 'FeedFwr_B')
        c.node('Add_in','Add')
    g.edge('Drop_in','FeedFwd_in_weight')
    g.edge('Drop_in','FeedFwd_in_bias')
    g.edge('FeedFwd_in_weight','Add_in')
    g.edge('FeedFwd_in_bias','Add_in')


    #Transformer nodes
    for layer in range(0,4):
        layer_name = f'net.transformer.layers.{layer}.'
        with g.subgraph(name = f'cluster_{layer}') as c:
            c.attr(label = layer_name)
            plot_transformer_model_layer(c, layer_name)

    #Output
    with g.subgraph(name = 'cluster_logit') as c:
        c.attr(label='Logit Layer')
        c.node('FeedFwd_out_weight1', 'FeedFwr_W')
        c.node('FeedFwd_out_bias1', 'FeedFwr_B')
        c.node('Add_out1','Add')
        c.node('Norm_out_weight','LayerNorm_W')
        c.node('Norm_out_bias', 'LayerNorm_B')
        c.node('Add_out2','Add')
        c.node('FeedFwd_out_weight2', 'FeedFwr_W')
        c.node('FeedFwd_out_bias2', 'FeedFwr_B')
        c.node('Add_out3','Add')
        c.node('ReLu_out1','ReLu')
        c.node('ReLu_out2','ReLu')
    g.edge('FeedFwd_out_weight1','Add_out1')
    g.edge('FeedFwd_out_bias1','Add_out1')
    g.edge('Add_out1','Norm_out_weight')
    g.edge('Add_out1','Norm_out_bias')
    g.edge('Norm_out_weight','Add_out2')
    g.edge('Norm_out_bias','Add_out2')
    g.edge('Add_out2','ReLu_out1')
    g.edge('ReLu_out1','FeedFwd_out_weight2')
    g.edge('ReLu_out1','FeedFwd_out_bias2')
    g.edge('FeedFwd_out_weight2','Add_out3')
    g.edge('FeedFwd_out_bias2','Add_out3')
    g.edge('Add_out3','ReLu_out2')

    #Probability
    with g.subgraph(name = 'cluster_prob') as c:
        c.attr(label = 'Probability Layer')
        c.node('FeedFwd_prob_weight1', 'FeedFwr_W')
        c.node('FeedFwd_prob_bias1', 'FeedFwr_B')
        c.node('Add_prob','Add')
        c.node('Sigmoid','Sigmoid')
    g.edge('FeedFwd_prob_weight1','Add_prob')
    g.edge('FeedFwd_prob_bias1','Add_prob')
    g.edge('Add_prob','Sigmoid')

    g.view()
    g.render(outfile = out_filename)

    return 0

if __name__ == "__main__":
    #plot_model("/Users/sveta/attn_classifier/")
    #generate_training_graph("/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/grads.pkl", '/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/grad_anim/')
    make_training_anim('/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/grad_anim/')
    #plot_scores_file('/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/scores.pkl')
    #plot_loss_file('/Users/sveta/attn_classifier/example_case_50pct_low_signal_good_chk_pt/train_output/loss.pkl')
    #plot_attention_maps_file("/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/maps.pkl", 999, 15)
    #plot_iteration_graph_from_file("/Users/sveta/attn_classifier/example_case_50pct_low_signal_good_chk_pt/train_output/grads.pkl", 0,'/Users/sveta/attn_classifier/example_case_50pct_low_signal_good_chk_pt/')
    #plot_param_grads_file("/Users/sveta/attn_classifier/example_case_10pct_low_signal_chk_pt/train_output/grads.pkl",
    #                      "net.transformer.layers.2.attn_layer.to_qvk.weight")