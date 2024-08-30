from FPSim2 import FPSim2Engine
import os
import wget
import sqlite3
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger
import warnings

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if not os.path.exists('data/chembl.h5'):
    # Download the file if it doesn't exist
    url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34.h5'
    wget.download(url, out='data/chembl.h5')
    
engine = FPSim2Engine(
    "data/chembl.h5",
)

def get_similar_compounds(canonical_smiles, engine):
    fps = engine.similarity(canonical_smiles, 0.7, n_workers=1)
    return fps

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

def process_compound(row, engine):
    fps = get_similar_compounds(row['canonical_smiles'], engine)
    similar_compounds = []
    for similar_compound in fps:
        similar_row = row.copy()
        similar_molregno = similar_compound[0]
        similar_row['similar_chembl_id'] = molregno_to_chembl.get(similar_molregno, f"Unknown_{similar_molregno}")
        similar_row['similarity_score'] = similar_compound[1]
        similar_compounds.append(similar_row)
    return similar_compounds

def process_batch(batch, engine):
    results = []
    for _, row in tqdm(batch.iterrows(), total=batch.shape[0]):
        results.extend(process_compound(row, engine))
    return results

def parallel_process_dataframe(df, n_processes=None, batch_size=1000):
    if n_processes is None:
        n_processes = cpu_count()
    
    engine = FPSim2Engine("data/chembl.h5")
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.starmap(process_batch, [(batch, engine) for batch in batches]),
            total=len(batches),
            desc="Processing batches",
            unit="batch"
        ))
    
    return [item for sublist in results for item in sublist]

# Use the function
similar_compounds = parallel_process_dataframe(
    compound_records_df,
    n_processes=12,  # Adjust based on your CPU cores
    batch_size=1000  # Adjust as needed
)

similar_compounds_df = pd.DataFrame(similar_compounds)
print(similar_compounds_df)

# Optionally, save the results to a CSV file
similar_compounds_df.to_csv('similar_compounds.csv', index=False)
