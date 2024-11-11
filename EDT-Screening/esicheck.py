import requests
import json
import pandas as pd
import time

# ChemSpider API settings
API_KEY = 'api_key'
BASE_URL = 'https://api.rsc.org/compounds/v1/filter'


# Function to get ChemSpider ID from IUPAC name
def get_chemspider_id(iupac_name):
    headers = {'apikey': API_KEY}
    payload = {
        "name": iupac_name,
        "order": "recordId",
        "include": ["recordId"]
    }
    response = requests.post(f"{BASE_URL}/name", headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        if data['recordCount'] > 0:
            return data['results'][0]  # Return the first matching ID
        else:
            print(f"No results found for {iupac_name}.")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None


# Function to get compound details from ChemSpider ID
def get_compound_info(chemspider_id):
    headers = {'apikey': API_KEY}
    response = requests.get(f"{BASE_URL}/record/{chemspider_id}/details", headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


# Function to check compatibility with ESI ionization (basic criteria)
def check_esi_compatibility(compound_info):
    # Example logic based on molecular weight; more criteria can be added
    mol_weight = compound_info.get('molecularWeight', 0)
    if mol_weight < 1000:  # Typically, larger molecules are less suited for ESI
        return "Likely compatible with ESI mode"
    else:
        return "Likely not compatible with ESI mode"


# Main function to process a list of IUPAC names from an Excel file
def main():
    # Load list of IUPAC names from Excel file
    df = pd.read_excel("list.xlsx")

    # Ensure the file has a column labeled 'IUPAC'
    if 'IUPAC' not in df.columns:
        print("Error: Excel file must contain an 'IUPAC' column.")
        return

    results = []

    for iupac_name in df['IUPAC']:
        print(f"Processing {iupac_name}...")
        chemspider_id = get_chemspider_id(iupac_name)

        if chemspider_id:
            print(f"Found ChemSpider ID: {chemspider_id}")
            compound_info = get_compound_info(chemspider_id)

            if compound_info:
                esi_result = check_esi_compatibility(compound_info)
                results.append({'IUPAC': iupac_name, 'ChemSpider ID': chemspider_id, 'ESI Compatibility': esi_result})
            else:
                results.append(
                    {'IUPAC': iupac_name, 'ChemSpider ID': chemspider_id, 'ESI Compatibility': 'Details not available'})
        else:
            results.append({'IUPAC': iupac_name, 'ChemSpider ID': 'Not found', 'ESI Compatibility': 'Not applicable'})

        # Optional delay to avoid API rate limits
        time.sleep(1)

    # Save results to a new Excel file
    output_df = pd.DataFrame(results)
    output_df.to_excel("esi_compatibility_results.xlsx", index=False)
    print("Results saved to esi_compatibility_results.xlsx")


# Call main function
main()
