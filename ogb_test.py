if __name__=="__main__":
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset

    from torch_geometric.data import DataLoader

    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    atom_encoder = AtomEncoder(emb_dim = 100)
    bond_encoder = BondEncoder(emb_dim = 100)

    

    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba')
    # dataset2 = PygPCQM4MDataset(root='./dataset')
    print(dataset[0])
    