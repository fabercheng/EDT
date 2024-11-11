# Event-Driven Mass Spectrometry Program

## Overview
This program is designed for the automated analysis of chemical compounds, focusing on generating Event-driven fingerprint (EDFP) and Event-driven ion (EDION) through advanced neural network models.

## Data
- Publicly available datasets and proprietary data accessible through scripts.
- Datasets include mass spectrometry data, chemical fingerprints, and other relevant chemical parameters.

## Additional Scripts
- **Data Acquisition (`pubchem_nist.py`)**: A web crawler script to fetch mass spectrometry data from PubChem.
- **MSP Compiler (`mspcompiler.R`)**: A script to extract and decode mass spectrometry information from commercial database files.
- **Configuration (`config.py`)**: Contains installation and configuration settings for the program.

## Program Structure
1. **ANN Model Architecture (`ann.py`)**: Defines the ANN for EDFP training.
2. **CCNN Model Architecture (`ccnn.py`)**: Outlines the CCNN for EDION training.

## EDFP Training Workflow

### 1. Data Preprocessing (`data_preprocessing.py`)
   - **Purpose**: Cleans and standardizes raw data.
   - **Process**: Load raw data, handle missing values, normalize features, and potentially reduce dimensions.

### 2. Feature Engineering (`feature_engineering.py`)
   - **Purpose**: Extracts meaningful features from the preprocessed data.
   - **Process**: Generate and select relevant chemical descriptors and fingerprints.

### 3. ANN Model Architecture (`ann.py`)
   - **Purpose**: Defines the structure and layers of the Artificial Neural Network for EDFP.
   - **Process**: Set up layers, activation functions, and compile the model.

### 4. Model Training (`model_training.py`)
   - **Purpose**: Trains the ANN model using the engineered features.
   - **Process**: Feed the data into the ANN model, adjust parameters, and initiate training.

### 5. Model Evaluation (`model_evaluation.py`)
   - **Purpose**: Assesses the performance of the trained model.
   - **Process**: Evaluate the model using metrics like accuracy, precision, recall, and F1 score.

### 6. Model Optimization (`model_optimization.py`)
   - **Purpose**: Fine-tunes the model for improved performance.
   - **Process**: Apply techniques like hyperparameter tuning to enhance the model.

### 7. Stacking (`stacking.py` and `qsar.py`)
   - **Purpose**: Integrates multiple models for robust predictions.
   - **Process**: Combine QSAR model to create a more powerful ensemble model.

### 8. EDFP Generation (`edfp.py`)
   - **Purpose**: Generates the Enhanced Digital Fingerprint using the trained ANN.
   - **Process**: Apply the ANN model to produce a comprehensive chemical fingerprint.

### 9. Applicability Domain (`applicability_domain.py`)
   - **Purpose**: Defines the domain where EDFP is reliable and performs data clustering.
   - **Process**: Use statistical methods to define EDFP's domain and reduce data dimensions.

### 10. EDFP Compiler (`edfp_compiler.py`)
   - **Purpose**: Annotates and compiles the structure of the generated EDFP.
   - **Process**: Analyze the EDFP data to identify key structural features.


## EDION Training Workflow

### 1. EDFP and HRMS Integration (`edfp_hrms.py`)
   - **Purpose**: Integrates EDFP results with HRMS data to prepare for CCNN training.
   - **Process**: Combines EDFP outputs with HRMS data for a comprehensive dataset.

### 2. CCNN Model Architecture (`ccnn.py`)
   - **Purpose**: Defines the structure of the Cascade Correlation Neural Network for EDION.
   - **Process**: Establishes CCNN layers, connections, and network parameters.

### 3. EDFP to EDION Transformation in GC-HRMS (`edfp_edion.py`)
   - **Purpose**: Transforms EDFP into EDION using the CCNN model.
   - **Process**: Trains the CCNN model with the curated dataset to generate EDION in EI modes.

### 4. Scoring System Definition (`scoring.py`)
   - **Purpose**: Develops a scoring system for compound analysis.
   - **Process**: Creates an algorithm that integrates EDION, mass spectrometry data, and retention indices.

### 5. Compound Screening (`screening.py`)
   - **Purpose**: Filters and selects compounds for analysis.
   - **Process**: Applies the EDION model to new datasets for compound screening.

### 6. EDFP to EDION Transformation in LC-HRMS (`edfp_edion_l.py`)
   - **Purpose**: Transforms EDFP into EDION using the CCNN model.
   - **Process**: Trains the CCNN model with the curated dataset to generate EDION in ESI modes.

## Installation and Usage
1. Clone or download the repository.
2. Follow instructions in `config.py` for setting up the environment.
3. Install necessary dependencies.
4. Execute the scripts as per the workflow for EDFP and EDION training.

## Version Updates
Modified the EDFP to EDION transformation module, added usage scripts for LC, and changed the original script's applicable object to GC.

## Notes
- Adherence to data usage and copyright regulations is essential.
- Results may vary based on datasets and parameter settings.

## Contact
- For inquiries or support, contact [fabercheng@icloud.com].
