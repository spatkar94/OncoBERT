import torch
import torch.nn as nn
import torch.nn.functional as F 

class BERT(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim=256, 
                 seq_len = 50, 
                 n_layers = 8, 
                 nheads = 8, 
                 use_plm = False,
                 plm_embeddings = None):
        super().__init__()

        if vocab_size is None:
                raise ValueError('Vocab size cannot be set to None.')

        self.use_plm = use_plm

        #learnable position embeddings
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        if self.use_plm:
            if plm_embeddings is None:
                raise ValueError('PLM embeddings cannot be empty if use_plm is set to True.')
            else:
                plm_emb_size = plm_embeddings.shape[1]
                self.register_buffer(name='plm_embeddings', tensor=plm_embeddings)
                self.val_emb = nn.Linear(plm_emb_size, embed_dim, bias=False)
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

        # logits layer to predict value of masked token
        self.val_logits = nn.Linear(embed_dim, vocab_size)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def save_checkpoint(self, epoch, loss=None, filename="best_model.pth"):
        """
        initialize model weights
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="best_model.pth", map_location=None):
        """
        load saved model weights
        """
        checkpoint = torch.load(filename, map_location=map_location)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)

        print(f"Loaded checkpoint from {filename} (epoch={epoch})")
        return epoch, loss

    def forward(self, seq, mask = None):
        B, T = seq.shape[0], seq.shape[1]

        # tokenization
        if self.use_plm:
            val_embeddings = self.val_emb(F.embedding(seq, self.plm_embeddings))
        else:
            val_embeddings = self.val_emb(seq) 

        # encoding token positions
        pos = torch.arange(0, T, device=seq.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(pos)
        
        # combine token values + position embeddings
        x = val_embeddings + pos_embeddings

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
        val_logits = self.val_logits(x[:,1:,:])
        class_token = x[:,0,:]


        return x[:,1:,:], val_logits, class_token

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MLPClassifier, self).__init__()
        # A MLP classifier
        self.classifier = nn.Sequential(*[nn.Linear(input_dim, input_dim),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.1),
                                          nn.Linear(input_dim, num_labels)])

    def forward(self, inputs):
        logits = self.classifier(inputs)
        return logits
    
    def save_checkpoint(self, epoch, loss=None, filename="best_model.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="best_model.pth", map_location=None):
        checkpoint = torch.load(filename, weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)

        print(f"Loaded checkpoint from {filename} (epoch={epoch})")
        return epoch, loss
    
