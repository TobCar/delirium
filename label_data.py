"""
@author: Tobias Carryer
"""

import pandas as pd

df_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")
df_labels = pd.read_csv