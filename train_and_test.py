import tensorflow as tf
import pandas as pd
from rdkit import Chem
import deepchem as dc
import numpy as np
import wget

keras = tf.keras
from graph_model import build_model

def download_data():
    wget.download("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv")  


def sanitize_molecule(smiles):
    """
    This function was taken from the Keras tutorial keras.io/examples/graph/mpnn-molecular-graphs/
    """
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def prepare_molecules(smiles_list):
    """
    This function extracts the atom features, the bond features, and the bond 
    pairs from each molecule.
    """
    atom_dataset, bond_dataset, pairs_dataset = [], [], []

    for smiles in smiles_list:
        molecule = sanitize_molecule(smiles)
        atoms, bonds, pairs = [], [], []

        # iterates over each atom in the molecule
        for atom in molecule.GetAtoms():

            # extracts features of length=78 from each atom
            atoms.append(dc.feat.graph_features.atom_features(atom, use_chirality=True))

            # iterates over each neighbor of each atom
            for neighbor in atom.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())

                # extracts features of length=10 from each bond
                bond_features = dc.feat.graph_features.bond_features(bond, use_chirality=True)
                bonds.append(np.array(bond_features, dtype=np.float32))

                # listing the pairs of atoms in a bond together
                pairs.append([atom.GetIdx(), neighbor.GetIdx()])

        atom_dataset.append(atoms)
        bond_dataset.append(bonds)
        pairs_dataset.append(pairs)

    # ragged tensors are used since the number of atoms and bonds are different in each molecule
    atom_dataset = tf.ragged.constant(atom_dataset, dtype=tf.float32)
    bond_dataset = tf.ragged.constant(bond_dataset, dtype=tf.float32)
    pairs_dataset = tf.ragged.constant(pairs_dataset, dtype=tf.int64)

    return (atom_dataset, bond_dataset, pairs_dataset)


def prepare_batch(inputs, labels):
    """
    This function combines all the molecules in the batch into a single graph.
    """
    atom_batch, bond_batch, pair_batch = inputs

    # number of atoms and bonds in each molecule
    # num_atoms and num_bonds shape = (batch_size,)
    num_atoms = atom_batch.row_lengths()
    num_bonds = bond_batch.row_lengths()

    # num_molecules = batch_size
    num_molecules = len(num_atoms)

    # merges the batch dimension and the number of atoms dimension
    atom_batch = atom_batch.merge_dims(0, 1).to_tensor()

    # merges the batch dimension and the number of bonds dimension
    bond_batch = bond_batch.merge_dims(0, 1).to_tensor()
    pair_batch = pair_batch.merge_dims(0, 1).to_tensor()

    # The following 3 steps add an increment to the bond pairs of a molecule where the increment 
    # equals the sum of the number atoms in the previous molecules in the batch. As a result, the 
    # indices of the pairs of each molecule don't conflict with the indices in other molecules.
    cumsum_atoms = tf.cumsum(num_atoms[:-1])
    increments = tf.concat([tf.constant([0], dtype=tf.int64), cumsum_atoms], axis=0)
    increments = tf.repeat(increments, num_bonds)
    pair_batch = pair_batch + tf.expand_dims(increments, axis=-1)

    # assigns a molecule_id to each atom so that we know which molecule they belong to
    # molecule_id shape = (total_num_atoms,)
    molecule_id = tf.repeat(tf.range(num_molecules), num_atoms)

    return (atom_batch, bond_batch, pair_batch, molecule_id), labels



def get_preprocess_fn(df, batch_size):
    """
    This function combines the inputs and labels in a single dataset and
    applies the preprocessing steps to the data.
    """
    def preprocess_fn(indices):
        index_df = df.iloc[indices]
        inputs = index_df.smiles
        inputs = prepare_molecules(inputs)
        labels = index_df.p_np

        dataset = tf.data.Dataset.from_tensor_slices((inputs, (labels)))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(prepare_batch, -1).prefetch(-1)

        return dataset
    
    return preprocess_fn


def get_dataset(batch_size):
    """
    This function splits the data into train, validation, and test datasets
    and applies the preprocessing steps to them.
    """

    df = pd.read_csv("BBBP.csv", usecols=[1, 2, 3])

    # Since the labels are imbalanced, the label with a lower frequency is 
    # given a higher weight and vice versa
    counts = np.bincount(df.p_np)
    label_weights = [1/counts[0], 1/counts[1]]

    # splitting the data into training, validation, and test datasets
    df_length = df.shape[0]
    random_indices = np.random.permutation(range(df_length))
    train_index = random_indices[ : int(df_length * 0.9)]
    valid_index = random_indices[int(df_length * 0.9) : int(df_length * 0.99)]
    test_index = random_indices[int(df_length * 0.99) : ]
    print("test_index: ", test_index)

    # applying the preprocessing steps to the datasets
    preprocess_fn = get_preprocess_fn(df, batch_size)
    train_dataset = preprocess_fn(train_index)
    valid_dataset = preprocess_fn(valid_index)
    test_dataset = preprocess_fn(test_index)

    return train_dataset, valid_dataset, test_dataset, label_weights


def run_training():
    """
    This script trains the graph model
    """
    batch_size = 20
    train_dataset, valid_dataset, test_dataset, label_weights = get_dataset(batch_size)

    model = build_model(batch_size)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.BinaryAccuracy()])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_binary_accuracy",
                                               patience=3,
                                               restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=1,
        class_weight={0: label_weights[0], 1: label_weights[1]},
        callbacks=[early_stop]
    )
    model.save_weights("graph_weights_1.h5")



def run_test():
    """
    This script tests the graph model
    """
    batch_size = 20
    df = pd.read_csv("BBBP.csv", usecols=[1, 2, 3])
    
    # test indices were saved when the dataset was split during training
    test_index = [215, 709, 1179, 486, 163, 1525, 776, 518, 360, 325, 681, 332, 1125, 798, 1688, 2034, 1306, 2006, 1140, 1076, 828]

    preprocess_fn = get_preprocess_fn(df, batch_size)
    test_dataset = preprocess_fn(test_index)

    new_model = build_model(batch_size)
    new_model.load_weights("graph_weights.h5")
    new_model.compile(loss=keras.losses.BinaryCrossentropy(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=[keras.metrics.BinaryAccuracy()])

    results = new_model.evaluate(test_dataset)[1]
    return results



