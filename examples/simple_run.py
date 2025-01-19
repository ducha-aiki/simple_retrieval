

from simple_retrieval.highlevel import SimpleRetrieval, get_default_config
import cv2
import argparse
import torch



def main(config=None, input_dir=None, global_desc_dir=None, local_desc_dir=None, query_fname=None):
    # Example usage of the retrieve_data function
    if config is None:
        config = get_default_config()
    r = SimpleRetrieval(img_dir=input_dir, index_dir= global_desc_dir, config=config)
    print (r)
    r.create_global_descriptor_index(input_dir,
                                     global_desc_dir)
    r.create_local_descriptor_index(input_dir)

    
    #r.create_global_descriptor_index('/Users/oldufo/datasets/oxford5k',
    #                                 './tmp/global_desc_ox5k')
    #r.create_local_descriptor_index('/Users/oldufo/datasets/oxford5k')
    #query_fname = '/Users/oldufo/datasets/oxford5k/all_souls_000006.jpg'
    #query_fname = '/Users/oldufo/datasets/EVD/1/graf.png'
    
    shortlist_idxs, shortlist_scores = r.get_shortlist(query_fname, num_nn=r.config["num_nn"])
    fnames = r.ds.samples  
    q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        idxs, scores = r.resort_shortlist(q_img, shortlist_idxs, matching_method=r.config["matching_method"],
                                 criterion=r.config["resort_criterion"], device=r.config["device"])
    print ([fnames[i] for i in idxs[:10]], scores[:10])  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple retrieval example')
    parser.add_argument('--input_dir', type=str, default='/Users/oldufo/datasets/goose',
                        help='Directory containing images')
    parser.add_argument('--query_fname', type=str, 
                        help='Query image filename')
    
    parser.add_argument('--global_desc_dir', type=str, default='./tmp/global_desc')
    parser.add_argument('--local_desc_dir', type=str, default='./tmp/local_desc')
    parser.add_argument('--resort_criterion', type=str, default='scale_factor_max', choices=['scale_factor_max', 'scale_factor_min', 'num_inliers'])	
    parser.add_argument('--num_nn', type=int, default=10)
    parser.add_argument('--inl_th', type=float, default=3.0)
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--force_recache', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--global_desc_batch_size', type=int, default=4)
    parser.add_argument('--local_desc_batch_size', type=int, default=4)
    parser.add_argument('--matching_method', type=str, default='snn')
    args = parser.parse_args()
    config = get_default_config()
    for k, v in vars(args).items():
        config[k] = v
    print (f"Config: {config}")
    main(config, args.input_dir, args.global_desc_dir, args.local_desc_dir, args.query_fname)