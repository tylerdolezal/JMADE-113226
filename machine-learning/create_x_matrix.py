from ase.io import vasp
import numpy as np
import glob

# Define a function to manually create one-hot encoding based on unique chemical symbols
def create_one_hot_encoder(unique_symbols):
    symbol_to_one_hot = {}
    num_symbols = len(unique_symbols)
    for i, symbol in enumerate(unique_symbols):
        one_hot = [0] * num_symbols
        one_hot[i] = 1
        symbol_to_one_hot[symbol] = one_hot
    return symbol_to_one_hot

# Initialize a shared mapping dictionary
# This accounts for adding different atomic species
symbol_mapping = {}
def assign_integer_labels(symbols):
    # Assign integer labels to symbols based on the shared mapping dictionary
    integer_labels = []
    for symbol in symbols:
        if symbol not in symbol_mapping:
            symbol_mapping[symbol] = len(symbol_mapping)+1
        integer_labels.append(symbol_mapping[symbol])
    
    symbol_to_one_hot = create_one_hot_encoder(sorted(set(symbols)))
    # Manually create one-hot encodings for atomic symbols
    atomic_types_encoded = np.array([symbol_to_one_hot[symbol] for symbol in symbols])
    
    return integer_labels, atomic_types_encoded


def extract_positions(dir):

    structures = glob.glob(dir+"/*")

    structures.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    # Read the CONTCAR file in direct coordinate

    X = []
    for afile in structures:
        atoms = vasp.read_vasp(afile)

        symbols = atoms.get_chemical_symbols()      
        if len(set(symbols)) != 3:
            print(afile)
        # Assign integer labels to symbols and account for newly added
        # atomic types
        integer_labels, encoded = assign_integer_labels(symbols)
        
        positions = atoms.positions
        #at_symbols = integer_labels
        vectors = atoms.cell.flatten()
        # Flatten atomic positions and one-hot encoded atomic types
        atomic_positions_encoded = np.hstack([positions, encoded]).flatten()
        
        data = np.concatenate((atomic_positions_encoded, vectors))
        #data = np.concatenate((vectors, atomic_positions_encoded))
        X.append(data)
        
    return(np.array(X))


def create_x_matrix(dir,save):
    #---------------------------------------------#
    # End of y matrix section #
    # Start building X matrix of characteristics #
    # energy, forces, stresses, EATOM, POMASS, and ZVAL
    #---------------------------------------------#
    # Begin with the LT phases #
    #---------------------------------------------#
    X= extract_positions(dir)
    
    X = np.array(X)
    print(X.shape)
    np.save(save,X)

path = "finished/structures"
save = "x-matrix.npy"

verify_path = "verification-structures/structures"
verify_save = "verification-x-matrix.npy"

mc2_path = "mc2-results/structures"
mc2_save = "mc2-x-matrix.npy"

new_path = "training_scripts/training_structures"
new_save = "training_scripts/new-x-matrix.npy"

create_x_matrix(path,save)
create_x_matrix(verify_path,verify_save)
#create_x_matrix(mc2_path,mc2_save)
create_x_matrix(new_path, new_save)
