import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sqlite3
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from rdkit import RDLogger
import warnings
import torch

FP_SIZE = 1024
RADIUS = 2
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Connect to the database
conn = sqlite3.connect('data/chembl_34.db')

query = """
    SELECT 
        md.molregno,
        md.chembl_id,
        md.pref_name,
        cs.canonical_smiles
    FROM 
        molecule_dictionary md
    LEFT JOIN
        compound_structures cs ON md.molregno = cs.molregno
    """

compound_records_df = pd.read_sql_query(query, conn)
print(compound_records_df)

'''
# load the model
ort_session = onnxruntime.InferenceSession(
    "trained_models/chembl_34_model/chembl_34_multitask.onnx", 
    providers=['CPUExecutionProvider']
)

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

def process_chunk(chunk, model_path, threshold=0.75):
    # Create InferenceSession inside the function
    ort_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    new_rows = []
    for _, row in chunk.iterrows():
        smiles = row['canonical_smiles']
        if pd.isna(smiles):
            continue
        
        try:
            descs = calc_morgan_fp(smiles)
            descs = descs.flatten()
            ort_inputs = {ort_session.get_inputs()[0].name: descs}
            preds = ort_session.run(None, ort_inputs)
            
            formatted_preds = format_preds(preds, [o.name for o in ort_session.get_outputs()])
            
            for target, score in formatted_preds:
                if score > threshold:
                    new_row = row.copy()
                    new_row['target'] = target
                    new_row['score'] = score
                    new_rows.append(new_row)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
    
    return pd.DataFrame(new_rows)

def parallel_process_dataframe(df, model_path, threshold=0.75, n_processes=None, chunk_size=1000):
    if n_processes is None:
        n_processes = 48
    
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(process_chunk, chunk, model_path, threshold) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Processing chunks"):
            results.append(future.result())
    
    return pd.concat(results, ignore_index=True)

# Use the function
model_path = "trained_models/chembl_34_model/chembl_34_multitask.onnx"
expanded_df = parallel_process_dataframe(
    compound_records_df, 
    model_path,
    threshold=0.75, 
    n_processes=32,  # Adjust based on your CPU cores
    chunk_size=2000  # Adjust based on your system's memory
)
print(expanded_df)

# Save the expanded dataframe to a CSV file
expanded_df.to_csv('data/expanded_df.csv', index=False)

