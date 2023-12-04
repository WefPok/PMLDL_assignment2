import pandas as pd
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero, GCNConv
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, RGCNConv, HeteroConv, GINConv
from torch_geometric.utils.dropout import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

data_folder = "../data/interim/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = nx.Graph()

np.random.seed(0)
pos = nx.spring_layout(G, seed=0)
torch.manual_seed(0)

users = pd.read_csv(data_folder + "users.csv")
items = pd.read_csv(data_folder + "items.csv")
ratings = pd.read_csv(data_folder + "ratings.csv")
genres = pd.read_csv("../data/raw/u.genre", delimiter="|", names=["name", "index"])

src = ratings["user_id"] - 1
dst = ratings["item_id"] - 1
attrs = ratings["rating"]

edge_index = torch.tensor([src, dst], dtype=torch.int64)
edge_attr = torch.tensor(attrs)


def SequenceEncoder(movie_titles, model_name=None):
    model = SentenceTransformer(model_name, device=device)
    title_embeddings = model.encode(movie_titles, show_progress_bar=True,
                                    convert_to_tensor=True, device=device)

    return title_embeddings.to("cpu")


item_title = SequenceEncoder(items["movie_title"], model_name='all-MiniLM-L6-v2')
item_genres = torch.tensor(items[genres.name].to_numpy(), dtype=torch.bool)
item_release_year = torch.tensor(items["release_year"].to_numpy()[:, np.newaxis], dtype=torch.int32)

item_x = torch.cat((item_title, item_genres), dim=-1).float()

occupations = [i for i in users.keys() if i.startswith("occupation_")]
user_ages = torch.tensor(users["age"].to_numpy()[:, np.newaxis], dtype=torch.uint8)
user_sex = torch.tensor(users[["male", "female"]].to_numpy(), dtype=torch.bool)
user_occupation = torch.tensor(users[occupations].to_numpy(), dtype=torch.bool)
user_x = torch.cat((user_ages, user_sex, user_occupation), dim=-1).float()
data = HeteroData()
data['user'].x = user_x
data['item'].x = item_x
data['user', 'rates', 'item'].edge_index = edge_index
data['user', 'rates', 'item'].edge_label = edge_attr
data = ToUndirected()(data)
del data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
data = data.to(device)
# Perform a link-level split into training, validation, and test edges.
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'item')],
    rev_edge_types=[('item', 'rev_rates', 'user')],
)(data)


class GNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), 32, add_self_loops=False)
        self.conv2 = GATv2Conv((-1, -1), 32, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['item'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder()
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        edge_label_index, mask = dropout_edge(edge_label_index, p=0.25, training=self.training)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index), mask


def evaluate(data):
    with torch.no_grad():
        model.eval()
        pred, _ = model(data.x_dict, data.edge_index_dict,
                        data['user', 'rates', 'item'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = data['user', 'rates', 'item'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        mae = torch.abs(target - pred).mean()
        return float(rmse), float(mae)


model = torch.load("../models/model.pt")
model.eval()

test_rmse, test_mae = evaluate(test_data)
print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")
