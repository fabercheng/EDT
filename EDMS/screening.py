import pandas as pd
from edfp_edion import generate_maccs_mass_string
from scoring import get_metfrag_scores
from scoring import calculate_retention_index_scores
from scoring import check_mass_spectrum_matches

def main():
    hrms_data = pd.read_csv('gc_hrms.csv')

    ei_data = pd.read_csv('ei.csv')

    generated_mass_list = generate_maccs_mass_string()

    for index, row in hrms_data.iterrows():
        sample_mass = row['mass']
        sample_ri = row['retention_index']

        metfrag_score = get_metfrag_scores()

        ri_score = calculate_retention_index_scores()

        edion_score = check_mass_spectrum_matches()

        total_score = 0.4 * metfrag_score + 0.3 * ri_score + 0.3 * edion_score

        print(f"Sample {index}: Total Score = {total_score}")

if __name__ == "__main__":
    main()
