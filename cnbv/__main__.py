import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--dataset_is_folder', action="store_true")
    parser.add_argument('--continuous_validation', action="store_true")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=400)
    parser.add_argument('--results_dir', type=str, default="/logs")
    parser.add_argument('--net', type=str, default="nbv-net")
    return parser
   
def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    
if __name__ == '__main__':
    main()