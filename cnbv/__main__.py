import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
        python -m cnbv --dataset=dataset --batch_size=16 --n_epochs=400 --results_dir=results_dir --net=nbvnet \
    --learning_rate=1e-4 --dropout_prob=0.3
    
    parser.add_argument('--dataset', type=str)
#     parser.add_argument('--dataset_is_folder', action="store_true")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default="/logs")
    parser.add_argument('--net', type=str, default="nbv-net")
    return parser

    
def main(args=None):
    #sys.argv
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    
if __name__ == '__main__':
    main()