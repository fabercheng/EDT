import tensorflow as tf
import numpy as np
import pandas as pd

# Load isotopic masses from a CSV file
def load_isotope_masses(filename):
    isotope_data = pd.read_csv(filename)
    isotope_masses = dict(zip(isotope_data['Element'], isotope_data['Mass']))
    return isotope_masses

# Initialize GPU memory growth to prevent full allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# LCMS-based class with ionization mode support
class CasCorNetModifiedWithLCMS:
    def __init__(self, input_size, output_size, args, isotope_masses, lcms_masses):
        # Initialize input and output sizes, arguments, and mass data
        self.input_size = input_size
        self.output_size = output_size
        self.args = args
        self.isotope_masses = isotope_masses  # Isotopic masses from CSV
        self.lcms_masses = lcms_masses  # Mass values from LCMS database
        self.I = input_size  # Current input size
        self.X_train = None  # Placeholder for training data
        self.X_test = None  # Placeholder for test data
        self.weights = None  # Placeholder for weights

    def augment_input(self, data, mass):
        # Append a new mass as a column to each row of the input data
        return np.hstack((data, np.full((data.shape[0], 1), mass)))

    def init_weights(self):
        # Initialize weights based on current input size and output size
        return np.random.rand(self.I, self.output_size)

    def add_hidden_unit_with_element_lcms(self, element, ion_mode='positive'):
        """
        Add an isotopic element as a hidden unit for LCMS with specified ionization mode.
        
        :param element: Element symbol to add
        :param ion_mode: Ionization mode ('positive' or 'negative')
        """
        if element not in self.isotope_masses:
            raise ValueError("Element not found in isotope masses")

        # Retrieve element mass and augment input data
        element_mass = self.isotope_masses[element]
        self.I += 1  # Increment input size
        self.X_train = self.augment_input(self.X_train, element_mass)
        self.X_test = self.augment_input(self.X_test, element_mass)

        # Reinitialize weights for the updated input size
        self.weights = self.init_weights()

        # Check for mass match in LCMS database based on ionization mode
        matched = self.check_lcms_match(ion_mode)
        if matched:
            print(f"Match found in LCMS library for {ion_mode} mode")
        else:
            print(f"No match found in LCMS library for {ion_mode} mode, continue adding elements")

    def check_lcms_match(self, ion_mode='positive'):
        """
        Check if the adjusted theoretical mass values match any LCMS mass based on ionization mode.
        
        :param ion_mode: Ionization mode ('positive' or 'negative')
        :return: True if a match is found, False otherwise
        """
        # Calculate theoretical mass from summed masses in the training set
        theoretical_masses = np.sum(self.X_train, axis=1)

        # Adjust theoretical mass for ionization mode
        if ion_mode == 'positive':
            theoretical_masses += 1.0073  # Proton mass adjustment for positive ionization
        elif ion_mode == 'negative':
            theoretical_masses -= 1.0073  # Proton mass adjustment for negative ionization
        else:
            raise ValueError("Invalid ionization mode. Use 'positive' or 'negative'.")

        # Check if any of the adjusted masses match those in the LCMS database
        return any(np.isclose(mass, theoretical_masses, atol=0.01) for mass in self.lcms_masses)

    def train(self, elements_to_add, ion_mode='positive'):
        """
        Training procedure for adding elements and checking mass matches in LCMS database.
        
        :param elements_to_add: List of elements to sequentially add during training
        :param ion_mode: Ionization mode ('positive' or 'negative')
        """
        for element in elements_to_add:
            print(f"Adding element {element} with ion mode {ion_mode}")
            self.add_hidden_unit_with_element_lcms(element, ion_mode)

# Load isotopic masses from the 'isotope.csv' file
isotope_masses = load_isotope_masses('isotope.csv')

# Define LCMS target masses (as example)
lcms_masses = [59.0073, 73.0073, 101.0073]  # Sample LCMS masses for positive ionization

# Define model parameters
input_size = 10
output_size = 5
args = {}

# Initialize LCMS model with specified parameters
model = CasCorNetModifiedWithLCMS(input_size, output_size, args, isotope_masses, lcms_masses)

# Initialize training and test data (replace with actual data loading in practice)
# For demonstration, we create random data to simulate training and testing sets
model.X_train = np.random.rand(100, input_size)  # 100 samples with input_size features
model.X_test = np.random.rand(50, input_size)    # 50 samples with input_size features

# List of elements to add during training process
elements_to_add = ['C', 'H', 'O', 'N']  # Define based on experimental needs

# Train model with the elements list in positive ionization mode
model.train(elements_to_add, ion_mode='positive')

