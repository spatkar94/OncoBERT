from oncobert.model import BERT, MLPClassifier
from oncobert.utils import *
import argparse
import anndata as ad

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script arguments")

    parser.add_argument("--mut_data", type=str,
                        help="Path to mutation data (saved in tabular csv format)")
    
    parser.add_argument('--plm_embeddings', type=str,
                        help='Path to h5 file storing ESM2 protein language model embeddings')
    
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint file where OncoBERT weights are saved")
    
    parser.add_argument("--save_loc", type=str, default='.',
                        help="Location where embeddings will be saved")
    
    parser.add_argument("--savename", type=str,
                        help="name of embeddings file (e.g., bert_embeddings.h5)")
    
    parser.add_argument("--context_length", type=int, default=50,
            help="sequence/context length (default: 50)")
    
    parser.add_argument("--embed_dim", type=int, default=256,
            help="embedding dimensions (default: 256)")
    
    parser.add_argument("--num_layers", type=int, default=8,
            help="number of transformer blocks (default: 8)")
    
    parser.add_argument("--classifier_chkpt", type=str,default=None,
            help="path to MLP classifier weights (default: None)")
    
    args = parser.parse_args()

    return vars(args)
    

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    mut_df = pd.read_csv(args['mut_data'], index_col=0)

    # Initialize dataset, model and training parameters
    mut_dataset = MutationDataset(mut_df, 
                                  plm_embedding_path=args['plm_embeddings'],
                                  context_length=args['context_length'], 
                                  mincount=1 
                                  )

    print(f'vocabulary size: {mut_dataset.df.shape[1]} \nnumber of sequences: {mut_dataset.df.shape[0]}')


    model = BERT(vocab_size=len(mut_dataset.features),
                        embed_dim=args['embed_dim'],
                        n_layers=args['num_layers'], 
                        seq_len= args['context_length'], 
                        use_plm=True, 
                        plm_embeddings=mut_dataset.feat_embeddings)
    
    model.load_checkpoint(filename=args['checkpoint'])

    device = torch.device('cuda:0')
    model = model.to(device)

    embeddings = gen_embeddings(dataset=mut_dataset,model=model, use_class_token=True)

    adata = ad.AnnData(embeddings)

    if args['classifier_chkpt'] is not None:
        mlp = MLPClassifier(256, 130)
        mlp = mlp.to(device)
        mlp.load_checkpoint(filename=args['classifier_chkpt'])
        # classify samples
        print("Classifying samples...")
        y_pred, prob = predict(adata.X, mlp, device)

        # save classification results
        adata.obs['classifier'] = y_pred

    #save as anndata file
    adata.write_h5ad(os.path.join(args['save_loc'],args['savename']))
    




    
