# Data Compilation for EDT-Screening
If you need, please download the HRMS spectral JSON file first. You can access it from open database, like MassBank and MoNA.
This project includes scripts:

1. **jsonread.py**  
   - Parses JSON files downloaded from the MassBank and MoNA databases.
   - Extracts and reads variable information from these files and retrieves ion details to support subsequent ionization mode verification.

2. **esicheck.py**  
   - Utilizes the ChemSpider API to verify ionization modes for a given list of substances based on their IUPAC names.
   - Automates the verification of Electrospray Ionization (ESI) compatibility for each substance, outputting the ionization mode results.


