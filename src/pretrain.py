from tqdm import trange
from utils import *
from model import Transformer
import torch
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


def train_one_epoch_mlm(model, optimizer, dl, mode, epoch, mask_fraction = 0.2):
    model.train()
    losses = []
    ce = torch.nn.CrossEntropyLoss()
    
    with trange(len(dl), desc="{}, Epoch {}: ".format(mode, epoch+1)) as t:
        for _, data in enumerate(dl):
            # 1. Zero the gradients
            optimizer.zero_grad()

            batch_size, seq_length = data.shape

            # move feature vals to GPU
            data = data.to(device)
            
            if model.feature_mode == 'categorical':
                tokens = data
            else:
                tokens = data.unsqueeze(-1).float()

            
            # targets for recovering feature names     
            feature_ids = torch.arange(seq_length).expand_as(data)

            # move to GPU
            feature_ids = feature_ids.to(device)

            # Number of tokens to mask per sequence (must be an integer)
            num_masked_tokens = int(seq_length*mask_fraction)  # → 74 (floor of 497 * 0.15)

            #random select tokens for masking
            mask_ids = torch.randint(0, seq_length, (batch_size, num_masked_tokens))
            mask_ids = mask_ids.to(device) 

            # run transformer model
            _, val_logits, _ = model(tokens, mask_ids)

            # collect masked token predictions and compute loss
            #batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(mask_ids)
            #masked_preds1 = feat_logits[batch_idx, mask_ids]
            #targets1 = torch.gather(feature_ids, dim=1, index=mask_ids)

            batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(mask_ids)
            masked_preds2 = val_logits[batch_idx, mask_ids]
            if model.feature_mode == 'categorical':
                targets2 = torch.gather(data, dim=1, index=mask_ids)
            else:
                targets2 = torch.gather(data.float(), dim=1, index=mask_ids)

            # compute loss
            #loss = ce(masked_preds1.transpose(1,2), targets1) + \
            #        ce(masked_preds2.transpose(1,2), targets2) \
            #            if model.feature_mode == 'categorical' \
            #            else mse(masked_preds2.squeeze(-1), targets2)
            
            loss = ce(masked_preds2.transpose(1,2), targets2)
            # 4. Backward pass: compute gradients
            loss.backward()

            # 5. Update model parameters
            optimizer.step()

            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()

    epoch_loss = sum(losses) / len(losses)
    print(f"Loss: {epoch_loss}")
    return epoch_loss

if __name__ == '__main__':
    # AACR GENIE
    muts = pd.read_csv('/data/sushantpa/AACR_GENIE/genie_chin_non_syn_muts_extended.csv', index_col=0)
    binary_df_large = muts.map(lambda x: 1 if x == 'MUT' else 0)
    source = np.array([x.split('-')[1] for x in binary_df_large.index])
    samps = np.array(list(binary_df_large.index))
    samps = np.array([x[10:] for x in samps])

    binary_df = pd.read_csv('/data/sushantpa/TCGA_tx/msk_chord_non_syn_muts.csv', index_col=0)
    binary_df = binary_df.map(lambda x: 1 if x == 'MUT' else 0)
    idx = ~np.isin(samps, list(binary_df.index))

    binary_df_large_filtered = binary_df_large[idx]

    print(f'Pretrain dataset. \nnumber of genes: {binary_df_large_filtered.shape[1]} \nnumber of samples: {binary_df_large_filtered.shape[0]}')


    # Initialize dataset, model and training parameters
    mut_dataset = MutationDataset(binary_df_large_filtered, context_length=50, mincount=1)
    
    print(f'Dataset size: {len(mut_dataset)}')
    print(f'Pretrain dataset. \nnumber of genes: {mut_dataset.df.shape[1]} \nnumber of samples: {mut_dataset.df.shape[0]}')

    print(f'Embedding dimension: {256}')
    
    model = Transformer(embed_dim=256,
                        n_layers=8, 
                        seq_len= mut_dataset.context_length,
                        vocab_size=len(mut_dataset.features), 
                        use_gpt=True, 
                        gpt_embeddings=mut_dataset.feat_embeddings)
    device = torch.device('cuda:0')
    model = model.to(device)
    num_epochs = 501
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    dl1 =  DataLoader(mut_dataset, batch_size=64, shuffle=True, num_workers=8)

    epoch_losses = []
    for epoch in range(num_epochs):
        eloss = train_one_epoch_mlm(model, optimizer, dl1, 'Train', epoch, mask_fraction=0.2)
        epoch_losses.append(eloss)
        if (epoch+1) % 100 == 0:
            model.save_checkpoint(epoch=(epoch+1), loss=eloss, filename=f'saved_checkpoints/best_model_epoch_{epoch}_no_msk_chord_extended_plm256.pth')
    print('Done Pretraining')
    

    # Create data frame saving epoch losses
    losses_df = pd.DataFrame({'epoch loss': epoch_losses})

    # Save plotting data to a file
    losses_df.to_csv('plots/epoch_losses.csv', index=False)