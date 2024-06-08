import torch
import torch.nn.functional as F
from torch import nn
from models.GraphEmbedding import *


class TS(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, mem_slots, head_size, output_size, num_heads=1,
                 num_blocks=1, num_layers=2,forget_bias=1., input_bias=0., dropout=0., layer_heads=[],batchnorm = True, num_edge_attr = 11,
                 embedding_model='GraphSage', interval=200,gate_style='unit', attention_mlp_layers=2, key_size=None, return_all_outputs=False):
        super(TS, self).__init__()
        # ------------Graph Embedding----------
        self.num_edge_attr = num_edge_attr
        self.edge_embeddings = nn.ModuleList([nn.Linear(num_edge_attr, hidden_size),nn.Linear(hidden_size, embedding_size)])
        self.interval = interval
        if embedding_model == 'GraphSage':
            self.embedding = GraphSage(input_size, hidden_size, embedding_size, num_layers, dropout,batchnorm)
        elif embedding_model == 'GAT':
            if not layer_heads:
                raise ValueError('layer heads is empty')
            self.embedding = GAT(input_size, hidden_size, embedding_size, num_layers, dropout, layer_heads,batchnorm)
        elif embedding_model == 'GATv2':
            if not layer_heads:
                raise ValueError('layer heads is empty')
            self.embedding = GATv2(input_size, hidden_size, embedding_size, num_layers, dropout, layer_heads,batchnorm)
        else:
            raise ValueError(
                'embedding model must be GraphSage, GATv2 or GAT. Your model type is {}'.format(embedding_model))
        ########## generic parameters for RMC ##########
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads

        # a new fixed params needed for pytorch port of RMC
        # +1 is the concatenated input per time step : we do self-attention with the concatenated memory & input
        # so if the mem_slots = 1, this value is 2
        self.mem_slots_plus_input = self.mem_slots + 1

        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(num_blocks))
        self.num_blocks = num_blocks

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. got: '
                '{}.'.format(gate_style))
        self.gate_style = gate_style

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers

        self.key_size = key_size if key_size else self.head_size

        ########## parameters for multihead attention ##########
        # value_size is same as head_size
        self.value_size = self.head_size
        # total size for query-key-value
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads  # denoted as F

        # each head has qkv_sized linear projector
        # just using one big param is more efficient, rather than this line
        # self.qkv_projector = [nn.Parameter(torch.randn((self.qkv_size, self.qkv_size))) for _ in range(self.num_heads)]
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.total_qkv_size])

        # used for attend_over_memory function
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        self.attended_memory_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])
        self.attended_memory_layernorm2 = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])

        ########## parameters for initial embedded input projection ##########
        self.embedding_size = embedding_size *2
        self.embedding_projector = nn.Linear(self.embedding_size, self.mem_size)

        ########## parameters for gating ##########
        self.num_gates = 2 * self.calculate_gate_size()
        self.input_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        # trainable scalar gate bias tensors
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

        ########## number of outputs returned #####
        self.return_all_outputs = return_all_outputs

        ######### output projector
        self.output_projector = nn.Linear(self.mem_slots*self.mem_size, output_size)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        # needed for truncated BPTT, called at every batch forward pass
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initial_state(self, batch_size, trainable=False):
        init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])

        # pad the matrix with zeros
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)

        # truncation. take the first 'self.mem_size' components
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :, :self.mem_size]

        return init_state

    def multihead_attention(self, memory):
        # First, a simple linear projection is used to construct queries
        qkv = self.qkv_projector(memory)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)

        # mem_slots needs to be dynamically computed since mem_slots got concatenated with inputs
        # example: self.mem_slots=10 and seq_length is 3, and then mem_slots is 10 + 1 = 11 for each 3 step forward pass
        # this is the same as self.mem_slots_plus_input, but defined to keep the sonnet implementation code style
        mem_slots = memory.shape[1]  # denoted as N

        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads, self.qkv_size)

        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)

        # [B, H, N, key_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        # scale q with d_k, the dimensionality of the key vectors
        q = q * (self.key_size ** -0.5)

        # make it [B, H, N, N]
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)

        # output is [B, H, N, V]
        output = torch.matmul(weights, v)

        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory

    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self.gate_style == None
            return 0

    def create_gates(self, inputs, memory):
        memory = torch.tanh(memory)

        # sonnet uses this, but i think it assumes time step of 1 for all cases
        # if inputs is (B, T, features) where T > 1, this gets incorrect
        # inputs = inputs.view(inputs.shape[0], -1)

        # fixed implementation
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1)
            # matmul for equation 4 and 5
            # there is no output gate, so equation 6 is not implemented
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        # this completes the equation 4 and 5
        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def attend_over_memory(self, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(memory)

            # Add a skip connection to the multiheaded attention's input.
            memory = self.attended_memory_layernorm(memory + attended_memory)

            # add a skip connection to the attention_mlp's input.
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)

        return memory

    def forward_step(self, inputs, memory, adjs, time, timestamp,edge_attr, treat_input_as_matrix=False):
        # ----------- Graph Embedding --------------
        device = inputs.device
        inputs,edge_index_time = self.embedding(inputs, adjs, time, timestamp, self.interval)
        # ----------- edge embedding -----------------
        edge_id = edge_index_time.storage.value().cpu()
        node_id = edge_index_time.storage.row().cpu()
        edge_attr =edge_attr[edge_id]
        edge_attr_x = torch.zeros(inputs.shape[0], self.num_edge_attr)
        edge_attr_x[node_id,edge_attr] = 1.0
        edge_attr_x = edge_attr_x.to(device)
        edge_attr_x = self.edge_embeddings[0](edge_attr_x)
        edge_attr_x = F.leaky_relu(edge_attr_x)
        edge_attr_x = self.edge_embeddings[1](edge_attr_x)
        # ----------- Temporal -----------------
        inputs_edge_attr = torch.cat([inputs, edge_attr_x], dim=-1)
        if treat_input_as_matrix:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            inputs_edge_attr = inputs_edge_attr.view(inputs_edge_attr.shape[0], inputs_edge_attr.shape[1], -1)
            # apply linear layer for dim 2
            inputs_reshape = self.embedding_projector(inputs_edge_attr)
        else:
            # keep (Batch, ...) dim (0), flatten starting from dim 1
            inputs_edge_attr =inputs_edge_attr.view(inputs.shape[0], -1)
            # apply linear layer for dim 1
            inputs_edge_attr = self.embedding_projector(inputs_edge_attr)
            # unsqueeze the time step to dim 1
            inputs_reshape = inputs_edge_attr.unsqueeze(dim=1)

        memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)
        next_memory = self.attend_over_memory(memory_plus_input)

        # cut out the concatenated input vectors from the original memory slots
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            # these gates are sigmoid-applied ones for equation 7
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            # equation 7 calculation
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory

        output = next_memory.view(next_memory.shape[0], -1)
        output = self.output_projector(output)
        return output, next_memory

    def forward(self, inputs, memory, adjs, timestamp, edge_attr, treat_input_as_matrix=False):
        logit = 0
        logits = []
        max_time = int(timestamp.max().item())
        # shape[1] is seq_lenth T
        for time in range(1, max_time - self.interval + 2, self.interval):
            #print([time + t for t in range(0, self.interval)])
            logit, memory = self.forward_step(inputs, memory, adjs, time, timestamp,edge_attr, treat_input_as_matrix)
            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)

        if self.return_all_outputs:
            return logits.log_softmax(dim=-1), memory
        else:
            return logit.log_softmax(dim=-1), memory


    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.embedding_projector.reset_parameters()
        self.qkv_projector.reset_parameters()
        self.qkv_layernorm.reset_parameters()
        for mlp in self.attention_mlp:
            mlp.reset_parameters()
        self.attended_memory_layernorm.reset_parameters()
        self.attended_memory_layernorm2.reset_parameters()
        self.input_gate_projector.reset_parameters()
        self.memory_gate_projector.reset_parameters()
        self.output_projector.reset_parameters()
