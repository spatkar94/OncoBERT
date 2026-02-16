import pandas as pd
from functools import reduce
import numpy as np
import glob
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
import h5py
import pickle
import os
device = torch.device('cuda:0')

#def save_esm_embeddings_to_h5py():
#    # Option 1 using PLM
#    files = glob.glob('/data/sushantpa/esm_embeddings/*.pt')
#    embeddings = {}
#    for i in tqdm(range(len(files))):
#        match = re.search(r'GN=([\w\d]+)', files[i])
#        if match:
#            gn_value = match.group(1)
#            embeddings[gn_value] = torch.load(files[i])['mean_representations'][33]
#    gene_embeddings = torch.stack([embeddings[gene] for gene in embeddings.keys()],dim=0)

#    with h5py.File('/data/sushantpa/TCGA_tx/esm2_data.h5', 'w') as f:
        # Save the PyTorch tensor
#        f.create_dataset('embeddings', data=gene_embeddings.cpu().numpy())

        # Save the Python list (by pickling it)
#        pickled_list = pickle.dumps(list(embeddings.keys()))
#        f.create_dataset('genes', data=np.void(pickled_list)) # Use np.void for arbitrary bytes


def prepare_input_data(mafpath, saveloc = '.', savename = 'geniev18_non_syn_muts_extended_test.csv'):
    df = pd.read_csv(mafpath,sep='\t')

    #keep only non-silent mutations
    df2 = df[df['Variant_Classification'].isin([
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'Nonstop_Mutation',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Splice_Site',
    'Translation_Start_Site'
    ])]

    mdf = df2[['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Classification']]
    mdf['status'] = 1

    # genes with at least 1 non-slient mutation in sample = 1
    # genes with no mutation/not profiled in sample = 0
    # not profiled and wt genes are grouped together for streamlining downstream data processing pipelines
    mut_matrix = mdf.pivot_table(index='Tumor_Sample_Barcode',columns='Hugo_Symbol',values='status',aggfunc='first',fill_value=0)
    
    # read in list of all profiled cancer associated genes from AACR GENIE
    target_genes = pd.read_csv('../data/profiled_genes_list.csv')['genes'].tolist()

    # reindex dataframe
    mut_matrix = mut_matrix.reindex(columns=target_genes, fill_value=0)

    print(f"Mutation Data Processed. Total: {mut_matrix.shape[0]} samples, {mut_matrix.shape[1]} genes")
    mut_matrix.to_csv(os.path.join(saveloc,savename),sep=",")
        
# create gene embeddings lookup table from ESM2 Protein Language Model
def get_plm_embeddings(genes, embeddings_path = 'data/mean_esm2_33M_embeddings.h5'):
    # Option 1 using PLM
    
    with h5py.File(embeddings_path, 'r') as f:
        embeddings_np = f['embeddings'][:]
        embeddings= torch.from_numpy(embeddings_np)
        all_genes = f['genes'][:].astype(str).tolist()
    common = np.intersect1d(genes, all_genes)
    idx = np.array([all_genes.index(gene) for gene in common])
    gene_embeddings = embeddings[idx,:]
    return common, gene_embeddings

# Option 2 create gene embeddings using GPT 3.5 applied to text descriptions of gene functions
def get_gpt3_5_embeddings(genes, gpt_embeddings_path):
    #gpt_embeddings_path = /data/sushantpa/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle
    with open(gpt_embeddings_path, "rb") as fp:
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


def rwr(adj, f0, restart_prob=0.5, max_iter=100, tol=1e-6):
    """
    run network diffusion algorithm implementation without symmetric normalization of node degrees
    
    :param adj: adjacency matrix capturing network topology and structure
    :param f0: intial state representing all mutated genes
    :param restart_prob: probability of random walk restart
    :param max_iter: max number of iterations
    :param tol: hyperparameter determining convergence
    """
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
    """
    run network diffusion algorithm implementation with symmetric normalization of node degrees
    
    :param adj: adjacency matrix capturing network topology and structure
    :param f0: intial state representing all mutated genes
    :param restart_prob: probability of random walk restart
    :param max_iter: max number of iterations
    :param tol: hyperparameter determining convergence
    """
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
    

    
class MutationDataset(Dataset):
    def __init__(self, X, context_length = 100, mincount = 5):
        super().__init__()
        self.df = X.copy()
        
        #minimum number of mutations per sample
        self.df = self.df[self.df.sum(axis=1) >= mincount]

       
        # extract gpt gene embeddings
        #common, feat_embeddings = get_gpt3_5_embeddings(list(self.df.columns))
        
        #print("Using plm embeddings")
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
        print("Transforming data into ordered sequences")
        self.sequences = {}
        
        for i in tqdm(range(self.df.shape[0])):
            x = torch.Tensor(self.df.T.iloc[:,i].values)
            f = rwr_symmetric(self.adj_matrix.to(device), x.to(device), restart_prob=0.5, max_iter=1000)
            seq = f.sort(descending=True)[1][:self.context_length].cpu().numpy()
            self.sequences[i] = seq

        print("Done!")
        
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        return self.sequences[index]

class EmbDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings)
        self.labels = labels
    
    def __len__(self):
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        return self.embeddings[idx,:], self.labels[idx]

def gen_embeddings(dataset, model, use_class_token = True):
    '''
    helper function to generate embeddings for a test dataset
    input: dataset <MutationDataset>
        model <BERT>
    output: ebeddings <numpy n x d > 
    '''


    embeddings = []
    model.eval()
    for i in tqdm(range(len(dataset))):
        eg = dataset[i]
        
        tens = torch.LongTensor(eg).unsqueeze(0)
        
        
        tens = tens.to(device)
        x, _, z = model(tens)
        if use_class_token:
            embeddings.append(z.detach().cpu().numpy())
        else:
            mean_embed = x.mean(dim=1)
            embeddings.append(mean_embed.detach().cpu().numpy())
    
    return np.vstack(embeddings)

def train_step(X_batch, y_batch, model, optimizer, criterion = nn.CrossEntropyLoss()):
    '''
    function to train MLP classifier on one minibatch
    '''
    
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(embeddings, model, device):
    '''
    function to predict OncoBERT subtype membership from embeddings
    
    :embeddings: OncoBERT generated embeddings
    :model: pretrained MLP classifier
    :device: GPU device
    '''
    embeddings = torch.from_numpy(embeddings)
    embeddings = embeddings.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(embeddings), dim=1)
    _, predicted_labels = torch.max(probs, 1)
    return predicted_labels.cpu().numpy(), probs.cpu().numpy()


