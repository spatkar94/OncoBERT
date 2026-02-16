from tqdm import trange
from oncobert.utils import *
from oncobert.model import BERT
import torch
from torch.utils.data import DataLoader
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')


def train_one_epoch_mlm(model, optimizer, dl, mode, epoch, mask_fraction = 0.2):
    model.train()
    losses = []
    ce = torch.nn.CrossEntropyLoss()
    
    with trange(len(dl), desc="{}, Epoch {}: ".format(mode, epoch+1)) as t:
        for _, tokens in enumerate(dl):
            # 1. Zero the gradients
            optimizer.zero_grad()

            batch_size, seq_length = tokens.shape

            # move tokens to GPU
            tokens = tokens.to(device)
            
            # Number of tokens to mask per sequence (must be an integer)
            num_masked_tokens = int(seq_length*mask_fraction)

            #random select tokens for masking
            mask_ids = torch.randint(0, seq_length, (batch_size, num_masked_tokens))
            mask_ids = mask_ids.to(device) 

            # run transformer model
            _, val_logits, _ = model(tokens, mask_ids)

            # collect masked token predictions and compute loss
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(mask_ids)
            masked_preds = val_logits[batch_idx, mask_ids]
            targets = torch.gather(tokens, dim=1, index=mask_ids)

            # compute masked token prediction loss
            loss = ce(masked_preds.transpose(1,2), targets)

            # Backward pass: compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()

    epoch_loss = sum(losses) / len(losses)
    print(f"Loss: {epoch_loss}")
    return epoch_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("--mut_data", type=str, default=None,
                        help="Path to mutation data (saved in tabular csv format)")
    
    parser.add_argument('--plm_embeddings', type=str,
                        help='Path to h5 file storing ESM2 protein language model embeddings')
    
    parser.add_argument("--save_loc", type=str, default=None,
                        help="Path to where model weights are saved")

    parser.add_argument("--num_epochs", type=int, default=500,
                        help="Number of training epochs (default: 500)")

    parser.add_argument("--context_length", type=int, default=50,
                        help="Sequence/context length (default: 50)")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")

    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (default: 256)")
    
    parser.add_argument("--num_layers", type=int, default=8,
            help="Number of Stacked Transformer layers (default: 8)")
    
    parser.add_argument("--num_workers", type=int, default=12,
            help='number of workers for data loader (default: 12)')
    
    parser.add_argument("--mask_fraction", type=float, default=0.2,
            help='fraction of tokens to be masked during training (default: 0.2)')
    
    
    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_args()
    set_seed(42)
    # AACR GENIE
    binary_df_large = pd.read_csv(args['mut_data'], index_col=0)
    source = np.array([x.split('-')[1] for x in binary_df_large.index])
    samps = np.array(list(binary_df_large.index))
    samps = np.array([x[10:] for x in samps])

    #subsampling for testing purposes only
    #idx = np.random.choice(binary_df_large.shape[0],size=10000)
    #binary_df_large = binary_df_large.iloc[idx,:]

    # Initialize dataset, model and training parameters
    mut_dataset = MutationDataset(binary_df_large,
                                  plm_embedding_path=args['plm_embeddings'],
                                  context_length=args['context_length'], 
                                  mincount=1
                                  )
    
    print(f'vocabulary size: {mut_dataset.df.shape[1]} \nnumber of sequences: {mut_dataset.df.shape[0]}')

    print("Initializing model")
    
    model = BERT(vocab_size=len(mut_dataset.features),
                        embed_dim=args['embed_dim'],
                        n_layers=args['num_layers'], 
                        seq_len= args['context_length'], 
                        use_plm=True, 
                        plm_embeddings=mut_dataset.feat_embeddings)
    device = torch.device('cuda:0')
    model = model.to(device)

    print(f"Embedding dimension: {args['embed_dim']}")


    num_epochs = args['num_epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9, 0.999), weight_decay=1e-4)
    dl1 =  DataLoader(mut_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

    epoch_losses = []
    for epoch in range(num_epochs):
        eloss = train_one_epoch_mlm(model, optimizer, dl1, 'Train', epoch, mask_fraction=args['mask_fraction'])
        epoch_losses.append(eloss)
        if (epoch+1) % 2 == 0:
            model.save_checkpoint(epoch=(epoch+1), loss=eloss, filename=os.path.join(args["save_loc"],f'oncobert_v{epoch}_{args["embed_dim"]}.pth'))
    print('Done training')
    

    # Create data frame saving epoch losses
    # losses_df = pd.DataFrame({'epoch loss': epoch_losses})

    # Save plotting data to a file
    # losses_df.to_csv(os.path.join(args['save_loc'],'epoch_losses_bert.csv'), index=False)
