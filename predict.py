import onnxruntime
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sqlite3
import pandas as pd
from multiprocessing import Pool, cpu_count
import multiprocessing.pool as mp
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

def process_row(row, model_path, threshold=0.75):
    # Create InferenceSession inside the function
    ort_session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
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

def parallel_process_dataframe(df, model_path, threshold=0.75, n_processes=None, batch_size=1000):
    if n_processes is None:
        n_processes = cpu_count()
    
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    available_gpus = get_available_gpus()
    n_gpus = len(available_gpus)
    
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.starmap(process_batch, [(batch, model_path, threshold, available_gpus[i % n_gpus]) for i, batch in enumerate(batches)]),
            total=len(batches),
            desc="Processing batches",
            unit="batch"
        ))
    
    return pd.concat(results, ignore_index=True)

def get_available_gpus():
    return list(range(torch.cuda.device_count()))

def process_batch(batch, model_path, threshold=0.75, gpu_id=0):
    # Set the CUDA device
    torch.cuda.set_device(gpu_id)
    
    # Create InferenceSession inside the function
    ort_session = onnxruntime.InferenceSession(
        model_path, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[{'device_id': str(gpu_id)}, {}]
    )
    
    smiles_list = batch['canonical_smiles'].tolist()
    valid_indices = [i for i, s in enumerate(smiles_list) if pd.notna(s)]
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    if not valid_smiles:
        return pd.DataFrame()
    
    new_rows = []
    for smiles in tqdm(valid_smiles, desc=f"Processing SMILES (GPU {gpu_id})", leave=False):
        descs = calc_morgan_fp(smiles)
        # Ensure the input is 1D
        descs = descs.flatten()
        ort_inputs = {ort_session.get_inputs()[0].name: descs}
        preds = ort_session.run(None, ort_inputs)
        
        formatted_preds = format_preds(preds, [o.name for o in ort_session.get_outputs()])
        
        for target, score in formatted_preds:
            if score > threshold:
                row = batch.iloc[valid_indices[valid_smiles.index(smiles)]]
                new_row = row.copy()
                new_row['target'] = target
                new_row['score'] = score
                new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

def parallel_process_dataframe(df, model_path, threshold=0.75, n_processes=None, batch_size=1000):
    if n_processes is None:
        n_processes = cpu_count()
    
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    available_gpus = get_available_gpus()
    n_gpus = len(available_gpus)
    
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_batch_wrapper, [
                (batch, model_path, threshold, available_gpus[i % n_gpus])
                for i, batch in enumerate(batches)
            ]),
            total=len(batches),
            desc="Processing batches",
            unit="batch"
        ))
    
    return pd.concat(results, ignore_index=True)

def process_batch_wrapper(args):
    return process_batch(*args)

# Use the function
model_path = "trained_models/chembl_34_model/chembl_34_multitask.onnx"
expanded_df = parallel_process_dataframe(
    compound_records_df, 
    model_path,
    threshold=0.75, 
    n_processes=48,  # Adjust based on your CPU cores
    batch_size=2000  # Adjust based on your GPU memory
)
print(expanded_df)

# Save the expanded dataframe to a CSV file
expanded_df.to_csv('data/expanded_df.csv', index=False)

