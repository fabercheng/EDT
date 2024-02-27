import os
import pandas as pd
from edfp_edion import generate_maccs_mass_string

metfrag_command = "java -jar MetFragCommandLine-VERSION-jar-with-dependencies.jar metfrag_parameters.txt"
os.system(metfrag_command)


def get_metfrag_scores():
    ei_data = pd.read_csv('ei.csv')
    return ei_data[['Compound', 'MetFragScore']]


def calculate_retention_index_scores():
    ei_data = pd.read_csv('ei.csv')
    ei_data['RIScore'] = ei_data['RI'].apply(lambda ri: 1 if abs(ri) <= 20 else 0)
    return ei_data[['Compound', 'RIScore']]


def check_mass_spectrum_matches(mass_values):
    tolerance = 0.001
    edion_masses = pd.DataFrame(mass_values, columns=['Mass'])
    edion_masses['MatchScore'] = edion_masses['Mass'].apply(lambda m: 1 if abs(m) <= tolerance else 0)
    return edion_masses
