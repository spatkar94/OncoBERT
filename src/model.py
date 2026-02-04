import torch
import torch.nn as nn
import torch.nn.functional as F 

class Transformer(nn.Module):
    def __init__(self, embed_dim=128, 
                 vocab_size = None, 
                 seq_len = 29, 
                 n_layers = 4, 
                 nheads = 8, 
                 use_gpt = False,
                 gpt_embeddings = None):
        super().__init__()

        self.feature_mode = 'categorical' if vocab_size is not None else 'numeric'
        self.use_gpt = use_gpt


        # encodes token names or positions (if set of tokens is ordered)
        self.feat_emb = nn.Embedding(seq_len, embed_dim)
        
        # encodes token values
        if self.feature_mode == 'numeric':
            self.val_emb = nn.Linear(1, embed_dim, bias=False)
        else:
            if self.use_gpt:
                if gpt_embeddings is None:
                    raise ValueError('GPT embeddings cannot be empty if use_gpt is set to True.')
                else:
                    gpt_emb_size = gpt_embeddings.shape[1]
                    self.register_buffer(name='gpt_embeddings', tensor=gpt_embeddings)
                    self.val_emb = nn.Linear(gpt_emb_size, embed_dim, bias=False)
            else:
                self.val_emb = nn.Embedding(vocab_size, embed_dim)

        
        

        # set up transformer blocks
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                                 nhead=nheads, 
                                                                 dropout=0.1, 
                                                                 batch_first = True) for _ in range(n_layers)])
        
        # create special class and mask tokens
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))

        # final normalization layer
        self.ln_f = nn.LayerNorm(embed_dim)

        # logits layer to predict identity/position of masked token
        self.feat_logits = nn.Linear(embed_dim, seq_len)

        # logits layer to predict value of masked token
        if self.feature_mode == 'numeric':
            self.val_logits = nn.Linear(embed_dim, 1)
        else:
            self.val_logits = nn.Linear(embed_dim, vocab_size)

        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def save_checkpoint(self, epoch, loss=None, filename="best_model.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, filename)
        print(f"✅ Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="best_model.pth", map_location=None):
        checkpoint = torch.load(filename, map_location=map_location)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)

        print(f"✅ Loaded checkpoint from {filename} (epoch={epoch})")
        return epoch, loss

    def forward(self, seq, mask = None):
        B, T = seq.shape[0], seq.shape[1]

        # encoding token values
        if self.feature_mode == 'numeric':
            val_embeddings = self.val_emb(seq) # B x T x C
        else:
            if self.use_gpt:
                val_embeddings = self.val_emb(F.embedding(seq, self.gpt_embeddings))
            else:
                val_embeddings = self.val_emb(seq) 

        # encoding token names/positions
        feat_ids = torch.arange(0, T, device=seq.device).unsqueeze(0)
        feat_embeddings = self.feat_emb(feat_ids)
        
        # combine token names/position and values
        x = feat_embeddings + val_embeddings

        # replace with mask token at masked locations if mask is not None
        if mask is not None:
            batch_idx = torch.arange(B).unsqueeze(1).expand_as(mask)
            x[batch_idx, mask] = self.mask_token


        #incorporate class token
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) 

        # pass appended seq to transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        #generate outputs
        feat_logits = self.feat_logits(x[:,1:,:])
        val_logits = self.val_logits(x[:,1:,:])
        class_token = x[:,0,:]


        return feat_logits, val_logits, class_token



class CoxSurv(nn.Module):
    def __init__(self, input_dim, out_dim, dropout = 0.2):
        super().__init__()

        self.feedforward = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, out_dim)
        ])

    def save_checkpoint(self, epoch, loss=None, filename="best_model.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, filename)
        print(f"✅ Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="best_model.pth", map_location=None):
        checkpoint = torch.load(filename, map_location=map_location)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)

        print(f"✅ Loaded checkpoint from {filename} (epoch={epoch})")
        return epoch, loss
    
    def forward(self, x):
        logits = self.feedforward(x)
        return logits
    
    
    