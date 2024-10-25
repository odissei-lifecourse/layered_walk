"Configurations for running layered walk codes"


config_dict = {
    "big": {
        "layers":  ["classmate", "household", "family", "colleague", "neighbor"],
        "walk_len": 50,
        "sample_size": 200_000
    },
    "small": {
        "layers": ["neighbor", "colleague"],
        "walk_len": 5,
        "sample_size": 10_000
    }
}

data_dir = {
    "snellius": {
        "input": "/projects/0/prjs1019/data/graph/processed/",
        "output": "/projects/0/prjs1019/data/graph/walks/"
    } ,
    "local": {
        "input": "/home/flavio/datasets/synthetic_layered_graph_1mil/",
        "output": "/home/flavio/datasets/synthetic_layered_graph_1mil/output"
    },
    "ossc": {
        "input": "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/",
        "output": "/gpfs/ostor/ossc9424/homedir/Dakota_network/random_walks/",
    }
}
