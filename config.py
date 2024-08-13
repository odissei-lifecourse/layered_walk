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
    "snellius": "/projects/0/prjs1019/data/graph/processed/",
    "local": "/home/flavio/datasets/synthetic_layered_graph_1mil/",
    "ossc": "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"
}
