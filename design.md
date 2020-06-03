dataset:
    1) folder. file-wise
    2) one file 
        (a) entire file to RAM
        (b) entire file to GRAM

usage:
    python -m cnbv --dataset=dataset --batch_size=16 --n_epochs=400 --results=results.csv --net_arch=nbv-net1 \
    --learning_rate=1e-4 --dropout_prob=0.3
    
the package is organised using init, submodules and subpackages
the package has __main__ file, so it can be run as a script too by 'python -m cnbv'