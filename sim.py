from FPSim2 import FPSim2CudaEngine
import os
import wget
import sqlite3
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from rdkit import RDLogger
import warnings
import cupy as cp

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if not os.path.exists('data/chembl.h5'):
    # Download the file if it doesn't exist
    url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34.h5'
    wget.download(url, out='data/chembl.h5')
    
engine = FPSim2CudaEngine(
    "data/chembl.h5",
)

def get_similar_compounds(canonical_smiles, engine):
    if pd.isna(canonical_smiles):
        return []  # Return an empty list if SMILES is NaN or None
    try:
        fps = engine.similarity(canonical_smiles, 0.7)
        return fps
    except Exception as e:
        print(f"Error processing SMILES: {canonical_smiles}. Error: {str(e)}")
        return []

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

# Create a dictionary to map molregno to chembl_id
molregno_to_chembl = dict(zip(compound_records_df['molregno'], compound_records_df['chembl_id']))

def process_batch(batch_data):
    batch, engine, molregno_to_chembl, gpu_id = batch_data
    
    # Set the GPU device for this process
    cp.cuda.Device(gpu_id).use()
    
    results = []
    for _, row in batch.iterrows():
        fps = get_similar_compounds(row['canonical_smiles'], engine)
        for similar_compound in fps:
            similar_row = row.copy()
            similar_molregno = similar_compound[0]
            similar_row['similar_chembl_id'] = molregno_to_chembl.get(similar_molregno, f"Unknown_{similar_molregno}")
            similar_row['similarity_score'] = similar_compound[1]
            results.append(similar_row)
    
    return results

def parallel_process_dataframe(df, engines, molregno_to_chembl, n_gpus=4, batch_size=1000):
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    batch_data = [
        (batch, engines[i % n_gpus], molregno_to_chembl, i % n_gpus)
        for i, batch in enumerate(batches)
    ]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = [executor.submit(process_batch, data) for data in batch_data]
        for future in tqdm(futures, total=len(batches), desc="Processing batches", unit="batch"):
            results.extend(future.result())
    
    return results

# Initialize engines for each GPU
engines = [FPSim2CudaEngine("data/chembl.h5", gpu_id=i) for i in range(4)]

# Use the function
similar_compounds = parallel_process_dataframe(
    compound_records_df,
    engines,
    molregno_to_chembl,
    n_gpus=4,  # Use all 4 GPUs
    batch_size=1000  # Adjust as needed
)

similar_compounds_df = pd.DataFrame(similar_compounds)
print(similar_compounds_df)

# Optionally, save the results to a CSV file
similar_compounds_df.to_csv('similar_compounds.csv', index=False)
