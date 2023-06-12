import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import inspect


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class GraphLayer(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        super(GraphLayer, self).__init__()

        self.linear1 = nn.Linear(in_features*2, hidden_features)
        self.elu1 = nn.ELU()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.elu2 = nn.ELU()

    def forward(self, h, edge_index):
        h = self._propagate(edge_index, h=h)
        return h

    def _propagate(self,edge_index, h):

        coll_dict = self.collect_dict(edge_index, h)

        # message computation
        msg_kwargs = {}
        for key in ['h_i', 'h_j']:
            _data = coll_dict.get(key, inspect.Parameter.empty)
            if _data is inspect.Parameter.empty:
                raise TypeError(f'Required parameter {key} is empty.')
            msg_kwargs[key] = _data

        out = self._message(**msg_kwargs)

        # message aggregation
        agg_kwargs = {}
        for key in ['index', 'dim_size', 'dim', 'reduce']:
            _data = coll_dict.get(key, inspect.Parameter.empty)
            if _data is inspect.Parameter.empty:
                raise TypeError(f'Required parameter {key} is empty.')
            agg_kwargs[key] = _data

        out = scatter(out, **agg_kwargs)

        # update
        out = out

        return out

    def _message(self, h_i, h_j):

        m = torch.cat([h_i, h_j], dim=-1)
        m = self.elu1(self.linear1(m))
        m = self.elu2(self.linear2(m))

        return m
    def collect_dict(self, edge_index, h):

        args = {'h_i','h_j'}
        i, j = (1, 0) # from source to target
        reduce_method = 'max'
        coll_dict = {}
        node_dim = -2
        num_nodes = h.shape[node_dim]

        for arg in args:
            dim = j if arg[-2:] == '_j' else i
            index = edge_index[dim]
            data = h.index_select(node_dim, index)
            coll_dict[arg] = data

        coll_dict['index'] = edge_index[i]
        coll_dict['dim_size'] = num_nodes
        coll_dict['reduce'] = reduce_method
        coll_dict['dim'] = node_dim

        return coll_dict

class GraphNet(nn.Module):
    def __init__(self, hidden_features, out_features):
        super(GraphNet, self).__init__()
        self.input_features = hidden_features
        self.hidden_features = hidden_features
        self.input_layer1 = nn.Linear(16, self.input_features)
        self.input_layer2 = nn.Linear(4, self.input_features)

        self.graph_layer1 = GraphLayer(self.input_features, hidden_features, hidden_features)
        self.graph_layer2 = GraphLayer(hidden_features, hidden_features, hidden_features)
        self.graph_layer3 = GraphLayer(hidden_features, hidden_features, out_features)


        self.edge_index = self.create_edge_index()

    def forward(self, input):
        object_feature, joint_feature = self.create_features(input) # input: [batch_size, num_dim:57]
        object_input = self.input_layer1(object_feature)
        joint_input = self.input_layer2(joint_feature).view(-1, 12, self.input_features)

        node_features = torch.cat([object_input.view(-1, 1, self.input_features),
                                   joint_input
                                   ], dim=1)

        h = self.graph_layer1(node_features, self.edge_index)
        h = self.graph_layer2(h, self.edge_index)
        h = self.graph_layer3(h, self.edge_index)

        return h

    def create_features(self,input):
        object_feature = torch.cat([input[:,0:10],input[:,10:16]],dim=1)

        joint_feature = torch.zeros((input.shape[0],12,4), dtype=torch.float32, device=device)

        for i in range(4):
            joint_feature[:,i,:] = torch.cat([input[:,i+16:i+16+1],input[:,i+16+12:i+16+12+1],input[:,i+16+24:i+16+24+1],input[:,i+16+36:i+16+36+1]],dim=1)

        for i in range(4):
            j = i * 2 +4
            joint_feature[:,i+4,:] = torch.cat([input[:,j+16:j+16+1],input[:,j+16+12:j+16+12+1],input[:,j+16+24:j+16+24+1],input[:,j+16+36:j+16+36+1]],dim=1)

        for i in range(4):
            j = i * 2 +5
            joint_feature[:,i+8,:] = torch.cat([input[:,j+16:j+16+1],input[:,j+16+12:j+16+12+1],input[:,j+16+24:j+16+24+1],input[:,j+16+36:j+16+36+1]],dim=1)


        # for i in range(4):
        #     joint_feature[:,i,:] = torch.cat([input[:,i+13:i+13+1],input[:,i+13+12:i+13+12+1],input[:,i+37:i+38]],dim=1)

        # for i in range(4):
        #     j = i * 2 +4
        #     joint_feature[:,i+4,:] = torch.cat([input[:,j+13:j+13+1],input[:,j+13+12:j+13+12+1],input[:,j+37:j+38]],dim=1)

        # for i in range(4):
        #     j = i * 2 +5
        #     joint_feature[:,i+8,:] = torch.cat([input[:,j+13:j+13+1],input[:,j+13+12:j+13+12+1],input[:,j+37:j+38]],dim=1)

        return object_feature, joint_feature

    def create_edge_index(self):
        # Prepare edge indices
        edge_index1 = torch.tensor([[0] * 4,
                                    list(range(1, 4 + 1))], dtype=torch.long, device=device)

        edge_index2 = torch.tensor([list(range(1, 4 + 1)),
                                    list(range(5, 4 + 5))], dtype=torch.long, device=device)

        edge_index3 = torch.tensor([list(range(5, 4 + 5)),
                                    list(range(9, 4 + 9))], dtype=torch.long, device=device)


        edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        edge_index = torch.cat([edge_index, torch.cat([edge_index[1:2, :], edge_index[0:1, :]], dim=0)], dim=1)

        # self.visualize_graph(12, edge_index)

        return edge_index

    def create_edge_index_new(self):
        # Prepare edge indices
        edge_index1 = torch.tensor([[0] * 4,
                                    list(range(9, 4 + 9))], dtype=torch.long, device=device)

        edge_index2 = torch.tensor([list(range(1, 4 + 1)),
                                    list(range(5, 4 + 5))], dtype=torch.long, device=device)

        # edge_index3 = torch.tensor([list(range(5, 4 + 5)),
        #                             list(range(9, 4 + 9))], dtype=torch.long, device=device)

        edge_index4 = torch.tensor([[0] * 4,
                                    list(range(5, 4 + 5))], dtype=torch.long, device=device)

        edge_index5 = torch.tensor([list(range(1, 4 + 1)),
                                    list(range(9, 4 + 9))], dtype=torch.long, device=device)

        edge_index = torch.cat([edge_index1, edge_index2,edge_index4,edge_index5], dim=1)
        edge_index = torch.cat([edge_index, torch.cat([edge_index[1:2, :], edge_index[0:1, :]], dim=0)], dim=1)

        # self.visualize_graph(12, edge_index)

        return edge_index

    def visualize_graph(self,num_nodes, edge_index):
        import networkx as nx
        import matplotlib.pyplot as plt

        # create an empty undirected graph
        G = nx.Graph()

        # add nodes to the graph
        num_nodes = num_nodes
        for i in range(num_nodes):
            G.add_node(i)

        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            u, v = edge_index[:, i].tolist()
            G.add_edge(u, v)

        # visualize the graph

        # position nodes using the kamada_kawai_layout method
        pos = nx.kamada_kawai_layout(G)

        # set the position of node 0 at the top and node 13 at the bottom
        min_y = min(y for x, y in pos.values())
        pos[0] = [0.0, 1.0]
        pos[num_nodes-1] = [pos[num_nodes-1][0], min_y]

        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.show()

class Action_Layer(nn.Module):
    def __init__(self, hidden_features, num_actions):
        super(Action_Layer, self).__init__()

        self.action_layer = nn.Linear(hidden_features, 1)
        self.num_actions = num_actions

    def forward(self, input):

        o = self.action_layer(input[:,1:13])

        return o.squeeze(-1)

class Value_Layer(nn.Module):
    def __init__(self, hidden_features):
        super(Value_Layer, self).__init__()

        self.hidden_features = hidden_features
        self.action_layer = nn.Linear(hidden_features, 1)

    def forward(self, input):

        input = torch.max(input,dim = 1).values

        o = self.action_layer(input)

        return o