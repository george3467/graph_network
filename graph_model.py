
import tensorflow as tf
keras = tf.keras
layers = keras.layers
import pandas as pd


class Dense_Layer(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.dense = layers.Dense(self.units)
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.2)
        
    def call(self, input):
        x = self.dense(input)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x



class Custom_GraphConvolution(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.dense_layer_1 = Dense_Layer(self.units)
        self.dense_layer_2 = Dense_Layer(self.units)
        self.dense_layer_3 = Dense_Layer(self.units)
        self.multiply = layers.Multiply()
        self.add = layers.Add()

    def call(self, atom_features, bond_features, bond_pairs):
        num_atoms = tf.shape(atom_features)[0]

        # atom_features shape = (num_atoms, units)
        atom_features = self.dense_layer_1(atom_features)

        # neighbor_atom_features shape = (num_bonds, units)
        neighbor_atom_features = tf.gather(atom_features, bond_pairs[:, 1])

        # neighbor_atom_features shape = (num_bonds, units)
        neighbor_atom_features = self.dense_layer_2(neighbor_atom_features)

        # bonds_features shape = (num_bonds, units)
        bonds_features = self.dense_layer_3(bond_features)

        # combined_features shape = (num_bonds, units)
        combined_features = self.multiply([neighbor_atom_features, bonds_features]) 

        # aggregate_neighbors shape = (num_atoms, units)
        aggregate_neighbors = tf.math.unsorted_segment_sum(combined_features,
                                                           bond_pairs[:, 0],
                                                           num_segments=num_atoms)
        # atom_features shape = (num_atoms, units)
        atom_features = self.add([atom_features, aggregate_neighbors])

        return atom_features



class Custom_GraphAttention(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units=units

    def build(self, input_shape):
        self.dense_layer_1 = Dense_Layer(self.units)
        self.dense_layer_2 = Dense_Layer(self.units)
        self.dense_layer_3 = Dense_Layer(units=1)
        self.multiply_1 = layers.Multiply()
        self.multiply_2 = layers.Multiply()
        self.multiply_3 = layers.Multiply()

    def call(self, atom_features, bond_features, bond_pairs):
        num_atoms = tf.shape(atom_features)[0]
        atom_ids = bond_pairs[:, 0]
        
        # neighbor_atom_features shape = (num_bonds, units)
        neighbor_atom_features = tf.gather(atom_features, bond_pairs[:, 1])

        # neighbor_atom_features shape = (num_bonds, units)
        neighbor_atom_features = self.dense_layer_1(neighbor_atom_features)

        # bond_features shape = (num_bonds, units)
        bond_features = self.dense_layer_2(bond_features)

        # combined_features shape = (num_bonds, units)
        combined_features = self.multiply_1([neighbor_atom_features, bond_features])

        # attention_weights shape = (num_bonds, 1)
        attention_weights = self.dense_layer_3(combined_features)

        # attention_weights shape = (num_bonds,)
        attention_weights = tf.squeeze(attention_weights, axis=-1)

        """
        The next six steps are based on the Keras Tutorial: 
        https://keras.io/examples/graph/gat_node_classification/
        """
        # The next 4 steps normalize the attention_weights        
        attention_weights = tf.math.exp(tf.clip_by_value(attention_weights, -2, 2))

        # sum_per_atom shape = (num_atoms,)
        sum_per_atom = tf.math.unsorted_segment_sum(attention_weights, 
                                                    segment_ids=atom_ids,
                                                    num_segments=tf.reduce_max(atom_ids)+1)

        # sum_per_atom_repeated shape = (num_bonds,)
        sum_per_atom_repeated = tf.repeat(sum_per_atom, tf.math.bincount(atom_ids))

        # attention_weights shape = (num_bonds, 1)
        attention_weights = tf.math.divide(attention_weights, sum_per_atom_repeated)[:, tf.newaxis]


        # weighted_neighbors shape = (num_bonds, units)
        weighted_neighbors = self.multiply_2([neighbor_atom_features, attention_weights])

        # weighted_atom_features shape = (num_atoms, units)
        weighted_atom_features = tf.math.unsorted_segment_sum(weighted_neighbors,
                                                                segment_ids=atom_ids,
                                                                num_segments=num_atoms)

        # weighted_bond_features shape = (num_bonds, units)
        weighted_bond_features = self.multiply_3([bond_features, attention_weights])
    
        return weighted_atom_features, weighted_bond_features



def build_model(batch_size):
    atom_features = keras.Input((78,), dtype=tf.float32)
    bond_features = keras.Input((10,), dtype=tf.float32)
    bond_pairs = keras.Input((2,), dtype=tf.int64)
    molecule_id = keras.Input((), dtype=tf.int32)

    # conv_atom shape = (num_atoms, 256)
    conv_atom = Custom_GraphConvolution(units=256)(atom_features, bond_features, bond_pairs)

    for i in range(4):
        # conv_atom shape = (num_atoms, 512)
        conv_atom = Custom_GraphConvolution(units=512)(conv_atom, bond_features, bond_pairs)
    
    # attn_atom shape = (num_atoms, 256); attn_bond shape = (num_bonds, 256)
    attn_atom, attn_bond = Custom_GraphAttention(units=256)(atom_features, 
                                                            bond_features,
                                                            bond_pairs)
    for i in range(4):
        # attn_atom shape = (num_atoms, 512); attn_bond shape = (num_bonds, 512)
        attn_atom, attn_bond  = Custom_GraphAttention(units=512)(attn_atom, 
                                                                 attn_bond,
                                                                 bond_pairs)
        
    # combined_features shape = (num_atoms, 1024)
    combined_features = keras.layers.Concatenate()([conv_atom, attn_atom])

    # sorted_features shape = (batch_size, 1024)
    sorted_features = tf.math.unsorted_segment_sum(combined_features,
                                                   segment_ids=molecule_id,
                                                   num_segments=batch_size)

    # x shape = (batch_size, 256)
    x = Dense_Layer(units=256)(sorted_features)

    # x shape = (batch_size, 128)
    x = Dense_Layer(units=128)(x)

    # output shape = (batch_size, 1)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model([atom_features, bond_features, bond_pairs, molecule_id], output)
    return model










