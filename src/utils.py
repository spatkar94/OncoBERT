import numpy
import pandas as pd
from functools import reduce
import os
import numpy as np
import glob
from tqdm import tqdm
import re
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import KFold
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss import cox
from model import *
from tqdm import trange
import h5py
import pickle
device = torch.device('cuda:0')

def save_esm_embeddings_to_h5py():
    # Option 1 using PLM
    files = glob.glob('/data/sushantpa/esm_embeddings/*.pt')
    embeddings = {}
    for i in tqdm(range(len(files))):
        match = re.search(r'GN=([\w\d]+)', files[i])
        if match:
            gn_value = match.group(1)
            embeddings[gn_value] = torch.load(files[i])['mean_representations'][33]
    gene_embeddings = torch.stack([embeddings[gene] for gene in embeddings.keys()],dim=0)

    with h5py.File('/data/sushantpa/TCGA_tx/esm2_data.h5', 'w') as f:
        # Save the PyTorch tensor
        f.create_dataset('embeddings', data=gene_embeddings.cpu().numpy())

        # Save the Python list (by pickling it)
        pickled_list = pickle.dumps(list(embeddings.keys()))
        f.create_dataset('genes', data=np.void(pickled_list)) # Use np.void for arbitrary bytes

def clean_and_preprocess_TCGA():
    # merge all tsv files
    df_list = []
    for i in range(13): 
        df = pd.read_csv(f'/data/sushantpa/TCGA_tx/tcga_mut_batch{i+1}.tsv', sep="\t")
        df = df.iloc[:,1:94]
        df.index = df['Sample ID']
        df.drop(columns=['Sample ID', 'Patient ID', 'Altered'], axis=1,inplace=True)
        df_list.append(df)

    df = pd.read_csv('/data/sushantpa/TCGA_tx/tcga_mut_batch14.tsv',sep="\t")
    df = df.iloc[:,1:18]
    df.index = df['Sample ID']
    df.drop(columns=['Sample ID','Patient ID', 'Altered'], axis=1,inplace=True)
    df_list.append(df)

    df_merged = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="inner"), df_list)

    # modify df to only record, alteration, no alteration, amp, del, remove any not profiled samples
    df_clean = df_merged[~df_merged.isin(['not profiled']).any(axis=1)]
    df_clean = df_clean.replace('no alteration','WT')

    # Allowed values
    allowed = ['WT','AMP','HOMDEL']

    # Replace any value not in allowed list with 'MUT'
    df_clean = df_clean.map(lambda x: x if x in allowed else 'MUT')

    df_clean['Sample ID'] = list(df_clean.index)
    df_clean.index = [x[:12] for x in df_clean.index]
    

    # read cancer types metadata
    ctypes = pd.read_csv('merged_ctypes.csv', index_col=0)

    # merge cancer type metadata with mutation data
    merged_data = pd.merge(df_clean, ctypes, left_index=True, right_index=True, how="inner")
    X = merged_data.drop(columns=['type'], axis=1)
    X.index = X['Sample ID']
    X.drop(columns=['Sample ID'], axis=1, inplace=True)
    Y = merged_data['type']

    return X, Y


#define the tokenizer

class Tokenizer:
    ''' 
    simple tokenizer that encodes specific mutations into integers
    '''
    def __init__(self, vocab):
        self.encode = {word:i for i, word in enumerate(vocab)}
        self.decode = {i:word for i, word in enumerate(vocab)}
    def tokenize(self, seq):
        return [self.encode[word] for word in seq]
    
    def detokenize(self, seq):
        return [self.decode[i] for i in seq]
        
# create gene embeddings lookup table from ESM2 Protein Language Model
def get_plm_embeddings(genes):
    # Option 1 using PLM
    with h5py.File('/data/sushantpa/TCGA_tx/esm2_data.h5', 'r') as f:
        embeddings_np = f['embeddings'][:]
        embeddings= torch.from_numpy(embeddings_np)

        all_genes_pickle = f['genes'][()] # Load as a scalar
        all_genes = pickle.loads(all_genes_pickle.tobytes())
    common = np.intersect1d(genes, all_genes)
    idx = np.array([all_genes.index(gene) for gene in common])
    gene_embeddings = embeddings[idx,:]
    return common, gene_embeddings

# Option 2 create gene embeddings using GPT 3.5 applied to text descriptions gene functions
def get_gpt3_5_embeddings(genes):
    
    with open('/data/sushantpa/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle', "rb") as fp:
        GPT_3_5_gene_embeddings = pickle.load(fp)

    for gene in GPT_3_5_gene_embeddings.keys():
        GPT_3_5_gene_embeddings[gene] = torch.tensor(GPT_3_5_gene_embeddings[gene])

    common = np.intersect1d(genes, list(GPT_3_5_gene_embeddings.keys()))
    gene_embeddings = torch.stack([GPT_3_5_gene_embeddings[gene] for gene in common],dim=0)
    return common, gene_embeddings

def mean_by_duplicate_columns(df):
    """Computes the mean of rows by duplicate column names in a Pandas DataFrame."""

    # Identify duplicate column names
    duplicated_columns = df.columns[df.columns.duplicated(keep=False)]

    # Create a new DataFrame with unique column names
    df_unique = df.loc[:, ~df.columns.duplicated()]

    # Calculate mean for each duplicate column group
    for col in duplicated_columns:
        df_unique[col] = df.loc[:, df.columns == col].mean(axis=1)

    return df_unique

def build_cosine_similarity_network(gene_embeddings: torch.Tensor, threshold: float = 0.7):
    """
    Builds a cosine similarity gene-gene interaction network.

    Args:
        gene_embeddings (torch.Tensor): Tensor of shape [num_genes, embedding_dim]
        threshold (float): Minimum cosine similarity to include an edge

    Returns:
        adjacency_matrix (torch.Tensor): Binary adjacency matrix [num_genes, num_genes]
        similarity_matrix (torch.Tensor): Cosine similarity matrix [num_genes, num_genes]
    """
    # Normalize the embeddings to unit vectors
    norm_embeddings = F.normalize(gene_embeddings, p=2, dim=1)

    # Compute cosine similarity matrix: sim[i,j] = cos(gene_i, gene_j)
    similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)

    # Zero diagonal (optional — remove self-similarity)
    similarity_matrix.fill_diagonal_(0.0)

    # Create binary adjacency matrix based on threshold
    adjacency_matrix = (similarity_matrix >= threshold).float()

    return adjacency_matrix, similarity_matrix




def visualize_gene_network(adjacency_matrix: torch.Tensor, gene_names=None, max_nodes=100, rwr_scores = None):
    """
    Visualize the gene-gene interaction network using NetworkX.

    Args:
        adjacency_matrix (torch.Tensor): Binary adjacency matrix [num_genes, num_genes]
        gene_names (list or None): Optional list of gene names (default: index-based labels)
        max_nodes (int): Max number of nodes to display (for clarity)
    """
    # Convert to numpy for NetworkX compatibility
    adj_np = adjacency_matrix.cpu().numpy()
    if rwr_scores is not None:
        scores_np = rwr_scores.cpu().numpy()
        scores_np = scores_np/np.max(scores_np)
    
    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adj_np)

    # Assign node attributes
    color_map = {}
    for i in G.nodes():
        color_map[i] = scores_np[i]

    # Normalize scores to colormap (0–1) and apply colormap
    cmap = cm.get_cmap('coolwarm')
    node_colors = [cmap(color_map[i]) for i in G.nodes()]


    # Limit number of nodes for visualization
    if G.number_of_nodes() > max_nodes:
        nodes_to_keep = list(range(max_nodes))
        G = G.subgraph(nodes_to_keep)
        node_colors = [node_colors[i] for i in nodes_to_keep]

    # Use gene names if provided
    if gene_names is not None:
        mapping = {i: gene_names[i] for i in G.nodes}
        G = nx.relabel_nodes(G, mapping)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # Generate layout for node positions
    pos = nx.kamada_kawai_layout(G)  # spring layout for readability

    # Draw the network
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=4)

    
    cbar = plt.colorbar(sm, label='RWR Score',ax=ax)

    plt.title("Gene-Gene Interaction Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def plot_heatmap(matrix: torch.Tensor, title="Matrix Heatmap", gene_names=None, cmap='inferno'):
    """
    Visualize a matrix (e.g., adjacency or similarity) as a heatmap.

    Args:
        matrix (torch.Tensor): 2D tensor [N, N]
        title (str): Title of the heatmap
        gene_names (list): Optional list of labels for axes
        cmap (str): Matplotlib colormap
    """
    matrix_np = matrix.cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix_np, aspect='auto', cmap=cmap)
    plt.colorbar(label='Value')
    if gene_names is not None and len(gene_names) <= 100:
        plt.xticks(range(len(gene_names)), gene_names, rotation=90, fontsize=6)
        plt.yticks(range(len(gene_names)), gene_names, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.title(title)
    plt.tight_layout()
    plt.show()


def rwr(adj, f0, restart_prob=0.5, max_iter=100, tol=1e-6):
    degree = adj.sum(dim=1)
    d_inv = torch.pow(degree, -1)
    d_inv[torch.isinf(d_inv)] = 0.0
    D_inv = torch.diag(d_inv).to(f0.device)
    normalized_adj = D_inv @ adj
    f = f0.clone().float()
    for _ in range(max_iter):
        f_new = (1 - restart_prob) * torch.matmul(normalized_adj, f) + restart_prob * f0
        if torch.norm(f_new - f, p=1) < tol:
            break
        f = f_new
    return f

def rwr_symmetric(adj, f0, restart_prob=0.5, max_iter=100, tol=1e-6):
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt).to(f0.device)
    normalized_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    f = f0.clone().float()
    for _ in range(max_iter):
        f_new = (1 - restart_prob) * torch.matmul(normalized_adj, f) + restart_prob * f0
        if torch.norm(f_new - f, p=1) < tol:
            break
        f = f_new
    return f

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def set_seed(seed):
	"""
	Utility function to set seed values for RNG for various modules
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False

def discretize(df, time_col,event_col,n_bins,eps = 1e-6):
    uncensored_df = df[df[event_col] > 0]
    #print(uncensored_df.shape)
    disc_labels, q_bins = pd.qcut(uncensored_df[time_col], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = df[time_col].max() + eps
    q_bins[0] = df[time_col].min() - eps
    disc_labels, q_bins = pd.cut(df[time_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    return disc_labels.values.astype(int)

    
class TranscriptomeDataset(Dataset):
    def __init__(self, X, nbins = None):
        super().__init__()
        self.df = X.copy()
        self.features = list(self.df.columns)
        self.nbins = nbins
        if self.nbins is not None:
            for col in self.features:
                self.df[col] = pd.cut(X[col], bins=nbins, labels=False, include_lowest=True)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        if self.nbins is not None:
            seq = torch.LongTensor(self.df.T.iloc[:,index].values)
        else:
            seq = self.df.T.iloc[:,index].values
        return seq
    
class MutationDataset(Dataset):
    def __init__(self, X, context_length = 100, mincount = 5):
        super().__init__()
        self.df = X.copy()
        
        #minimum number of mutations per sample
        self.df = self.df[self.df.sum(axis=1) >= mincount]

       
        # extract gpt gene embeddings
        #common, feat_embeddings = get_gpt3_5_embeddings(list(self.df.columns))
        print("using plm embeddings")
        common, feat_embeddings = get_plm_embeddings(list(self.df.columns))

        #filter out genes without embeddings
        self.df = self.df[common]

        # feature names
        self.features = common

        # gpt gene embeddings
        self.feat_embeddings = feat_embeddings

        #construct gene interaction network
        adj, sim = build_cosine_similarity_network(self.feat_embeddings, threshold=0.9)
        self.adj_matrix = adj
        #print("using similarity matrix")

        # define max context length
        self.context_length = context_length

        # run rwr
        print("Data preprocessing... Running network propagation")
        self.sequences = {}
        
        for i in tqdm(range(self.df.shape[0])):
            x = torch.Tensor(self.df.T.iloc[:,i].values)
            f = rwr_symmetric(self.adj_matrix.to(device), x.to(device), restart_prob=0.5, max_iter=1000)
            seq = f.sort(descending=True)[1][:self.context_length].cpu().numpy()
            self.sequences[i] = seq
        
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        return self.sequences[index]

class SurvDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return (self.X.iloc[index,:].values, 
                self.Y['status'].iloc[index],
                self.Y['time'].iloc[index])
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def gen_embeddings(dataset, model):
    '''
    helper function to generate embeddings for a test dataset
    input: dataset <MutationDataset>
        model <Transformer>
    output: ebeddings <numpy n x d > 
    '''


    embeddings = []
    model.eval()
    for i in tqdm(range(len(dataset))):
        eg = dataset[i]
        if model.feature_mode == 'categorical':
            tens = torch.LongTensor(eg).unsqueeze(0)
        else:
            tens = torch.tensor(eg).unsqueeze(0).unsqueeze(-1).float()
        
        tens = tens.to(device)
        _, _, z = model(tens)
        embeddings.append(z.detach().cpu().numpy())
    
    return np.vstack(embeddings)

def create_cv_splits(X,Y, n_splits = 5):
    train_val_splits = {}
    # Create a KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterate over the splits
    for fold, (train_index, val_index) in enumerate(kf.split(X = X)):
        train_val_splits[fold] = {}

        train_val_splits[fold]['train'] = {}
        train_val_splits[fold]['train']['X'] = X.iloc[train_index,:]
        train_val_splits[fold]['train']['Y'] = Y.iloc[train_index,:]

        train_val_splits[fold]['val'] = {}
        train_val_splits[fold]['val']['X'] = X.iloc[val_index,:]
        train_val_splits[fold]['val']['Y'] = Y.iloc[val_index,:]
    
    return train_val_splits

def train_one_epoch(model, surv_mod, optimizer, dataloader):
    model.train()
    surv_mod.train()
    losses = []
    
    for _, (X, c, time) in enumerate(dataloader):

        optimizer.zero_grad()

        X = X.to(device)
        c = c.to(device)
        time = time.to(device)

        _,_,z = model(X)
        hazards = surv_mod(z)

        loss = cox.neg_partial_log_likelihood(hazards, c, time)
        loss_value = loss.item()
        
        #backprop
        loss.backward()

        #updtae parameters
        optimizer.step()

        losses.append(loss_value)
    
    epoch_loss = sum(losses) / len(losses)

    return epoch_loss

def val_one_epoch(model, surv_mod, dataloader):
    model.eval()
    surv_mod.eval()
    losses = []
    risk_scores = []
    events = []
    event_times = []
    for _, (X, c, time) in enumerate(dataloader):

        X = X.to(device)
        c = c.to(device)
        time = time.to(device)

        _,_,z = model(X)
        hazards = surv_mod(z)

        loss = cox.neg_partial_log_likelihood(hazards, c, time)
        loss_value = loss.item()
        losses.append(loss_value)

        
        risk_scores.append(hazards.squeeze(0).detach().cpu().numpy())
        events.append(c.detach().cpu().numpy())
        event_times.append(time.detach().cpu().numpy())

    epoch_loss = sum(losses) / len(losses)
    events = torch.tensor(np.concatenate(events)) > 0
    event_times = torch.tensor(np.concatenate(event_times))
    risk_scores = torch.tensor(np.concatenate(risk_scores))

    cindex = ConcordanceIndex()
    cind = cindex(risk_scores, events, event_times)

    return epoch_loss, cind


class Trainer:
    def __init__(self, embedding_model, num_epochs=15, use_pretrained=True):
        self.num_epochs = num_epochs
        self.embedding_model = embedding_model.to(device)

        if use_pretrained:
            self.embedding_model.load_checkpoint(filename='saved_checkpoints/best_model_epoch_499_no_msk_chord_extended.pth')
        
        self.surv_model = CoxSurv(input_dim=128, out_dim=4)
        self.surv_model = self.surv_model.to(device)
        
        
    def fit(self, train_dataset, val_dataset = None, batch_size = 32):
        

        train_loader = DataLoader(train_dataset, 
                                  num_workers=8, 
                                  batch_size=batch_size, 
                                  shuffle=True)
        
        val_loader = DataLoader(val_dataset, 
                                  num_workers=8, 
                                  batch_size=batch_size, 
                                  shuffle=False)


        # create optimizer
        optimizer = torch.optim.AdamW(list(list(self.embedding_model.parameters()) + list(self.surv_model.parameters())), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

        best_train_loss, best_val_loss, best_c_index = float("inf"), float("inf"), 0.0
        best_valid_epoch = 0
        early_stopper = EarlyStopper(patience=10, min_delta=0.0)

        for epoch in tqdm(range(self.num_epochs)):
            train_loss = train_one_epoch(self.embedding_model, self.surv_model,  optimizer, train_loader)
            val_loss, c_index = val_one_epoch(self.embedding_model, self.surv_model, val_loader)
            print("C-index at epoch {}: {:.3f}".format(epoch, c_index))
            
        self.embedding_model.save_checkpoint(best_valid_epoch, loss=best_val_loss, filename=f"finetuned_emb_model.pth")
        self.surv_model.save_checkpoint(best_valid_epoch, loss=best_val_loss, filename=f"finetuned_surv_model.pth")

            

    def evaluate(self, test_dataset, batch_size = 32):
        val_loader = DataLoader(test_dataset, 
                                  num_workers=8, 
                                  batch_size=batch_size, 
                                  shuffle=False)
        val_loss, c_index = val_one_epoch(self.embedding_model, self.surv_model, val_loader)
        return c_index

            
def train_surv_model(X,Y, model, num_epochs, use_pretrained = True, num_workers = 8, batch_size = 32, learning_rate = 1e-4, checkpoints_file = 'saved_checkpoints/best_model_epoch_199_no_msk_chord_extended_2.pth'):
    
    train_val_splits = create_cv_splits(X, Y, n_splits=5)
    c_indices = []
    for fold in range(5):
        print(f'starting fold: {fold}')

        #initialize models
        if use_pretrained:
            model.load_checkpoint(filename=checkpoints_file)
        
        model = model.to(device)
        surv_model = CoxSurv(input_dim=128, out_dim=1)
        surv_model = surv_model.to(device)


        # create optimizer
        optimizer = torch.optim.AdamW(list(list(model.parameters()) + list(surv_model.parameters())), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

        data_x = train_val_splits[fold]['train']['X']
        data_y = train_val_splits[fold]['train']['Y']
        train_loader = DataLoader(SurvDataset(data_x, data_y), 
                                  num_workers=num_workers, 
                                  batch_size=batch_size, 
                                  shuffle=True)
        
        data_x = train_val_splits[fold]['val']['X']
        data_y = train_val_splits[fold]['val']['Y']
        val_loader = DataLoader(SurvDataset(data_x, data_y), 
                                  num_workers=num_workers, 
                                  batch_size=batch_size, 
                                  shuffle=False)
        

        best_train_loss, best_val_loss, best_c_index = float("inf"), float("inf"), 0.0
        best_valid_epoch = 0
        early_stopper = EarlyStopper(patience=10, min_delta=0.0)

        for epoch in tqdm(range(num_epochs)):
            train_loss = train_one_epoch(model, surv_model,  optimizer, train_loader)
            val_loss, c_index = val_one_epoch(model, surv_model, val_loader)
        

            if val_loss < best_val_loss:
                best_epoch, best_train_loss, best_val_loss, best_c_index = epoch, train_loss, val_loss, c_index
                best_valid_epoch = best_epoch
                #model.save_checkpoint(best_valid_epoch, loss=best_val_loss, filename=f"best_surv_model_fold_{fold}.pth")

            if early_stopper.early_stop(val_loss):
                print("#### Early stopped...")
                c_indices.append(best_c_index)
                break
        
        print(f"Epoch for best validation loss : {best_valid_epoch} \n")
        print("Train Loss at epoch {} (best model): {:.3f}".format(best_valid_epoch, best_train_loss))
        print("Val Loss at epoch {} (best model): {:.3f}".format(best_valid_epoch, best_val_loss))
        print("C-index at epoch {} (best model): {:.3f}".format(best_valid_epoch, best_c_index))

    return np.mean(c_indices)
    


        

