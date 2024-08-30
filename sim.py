from FPSim2 import FPSim2Engine
import os
import wget
import sqlite3
import pandas as pd

if not os.path.exists('data/chembl.h5'):
    # Download the file if it doesn't exist
    url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34.h5'
    wget.download(url, out='data/chembl.h5')
    
engine = FPSim2Engine(
    "data/chembl.h5",
)

query = "CC(=O)OC1=CC=CC=C1C(=O)O"
fps = engine.similarity(query, 0.7, n_workers=12)
print(fps)

def get_similar_compounds(row):
    engine = FPSim2Engine(
        "data/chembl.h5",
    )
    fps = engine.similarity(row['canonical_smiles'], 0.7, n_workers=12)
    return fps

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
print(compound_records_df)

for fp in fps:
    print(fp)
    # find the row where molregno matches the fp
    row = compound_records_df[compound_records_df['molregno'] == fp[0]]
    print(row)
    print()
