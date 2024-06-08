# model parameters
hidden_size = 512
embedding_size = 256
mem_slots = 16
head_size = 128
num_heads = 1
num_blocks =1
num_layers = 2
forget_bias = 1.0
input_bias = 0.
dropout = 0.
embedding_model = "GraphSage" # "GAT" "GATv2"
layer_heads = [4,1]
batchnorm = True
interval = 300
gate_style = 'unit'
attention_mlp_layers=2
key_size = None # if None, use head_size * num_heads
return_all_outputs=False
lr = 0.00008
l2 = 5e-7
num_edge_attr = 11
#