import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sqlite3
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing.pool as mp
from tqdm import tqdm

FP_SIZE = 1024
RADIUS = 2

def calc_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=FP_SIZE)
    a = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return a

def format_preds(preds, targets):
    preds = np.concatenate(preds).ravel()
    np_preds = [(tar, pre) for tar, pre in zip(targets, preds)]
    dt = [('chembl_id','|U20'), ('pred', '<f4')]
    np_preds = np.array(np_preds, dtype=dt)
    np_preds[::-1].sort(order='pred')
    return np_preds

# load the model
ort_session = onnxruntime.InferenceSession(
    "trained_models/chembl_34_model/chembl_34_multitask.onnx", 
    providers=['CPUExecutionProvider']
)


# Connect to the database
conn = sqlite3.connect('data/chembl_34.db')

query = """
    SELECT 
        md.molregno,
        md.chembl_id,
        md.pref_name,
        cr.compound_name,
        cs.canonical_smiles
    FROM 
        molecule_dictionary md
    LEFT JOIN 
        compound_records cr ON md.molregno = cr.molregno
    LEFT JOIN
        compound_structures cs ON md.molregno = cs.molregno
    """

compound_records_df = pd.read_sql_query(query, conn)
print(compound_records_df.head())

'''
# calculate the FPs
smiles = 'CN(C)CCc1c[nH]c2ccc(C[C@H]3COC(=O)N3)cc12'
descs = calc_morgan_fp(smiles)

# run the prediction
ort_inputs = {ort_session.get_inputs()[0].name: descs}
preds = ort_session.run(None, ort_inputs)

# example of how the output of the model can be formatted
preds = format_preds(preds, [o.name for o in ort_session.get_outputs()])
print(preds)
'''

def process_row(row, model_path, threshold=0.75):
    # Create InferenceSession inside the function
    ort_session = onnxruntime.InferenceSession(
        model_path, 
        providers=['CPUExecutionProvider']
    )
    
    smiles = row['canonical_smiles']
    if pd.isna(smiles):
        return pd.DataFrame()  # Return empty DataFrame if SMILES is NaN
    
    descs = calc_morgan_fp(smiles)
    # Ensure the input is 1D
    descs = descs.flatten()
    ort_inputs = {ort_session.get_inputs()[0].name: descs}
    preds = ort_session.run(None, ort_inputs)
    
    formatted_preds = format_preds(preds, [o.name for o in ort_session.get_outputs()])
    
    # Filter predictions above threshold and create new rows
    new_rows = []
    for target, score in formatted_preds:
        if score > threshold:
            new_row = row.copy()
            new_row['target'] = target
            new_row['score'] = score
            new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

def process_row_wrapper(args):
    return process_row(*args)

def parallel_process_dataframe(df, model_path, threshold=0.75, n_processes=10):
    if n_processes is None:
        n_processes = cpu_count()
    
    with Pool(n_processes) as pool:
        total = len(df)
        results = list(tqdm(
            pool.imap(process_row_wrapper, [(row, model_path, threshold) for _, row in df.iterrows()]),
            total=total,
            desc="Processing compounds",
            unit="compound"
        ))
    
    return pd.concat(results, ignore_index=True)

# Use the function
model_path = "trained_models/chembl_34_model/chembl_34_multitask.onnx"
expanded_df = parallel_process_dataframe(
    compound_records_df, 
    model_path,
    threshold=0.75, 
    n_processes=10
)
print(expanded_df)

# Save the expanded dataframe to a CSV file
expanded_df.to_csv('data/expanded_df.csv', index=False)
