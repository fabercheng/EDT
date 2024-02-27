import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class CasCorNetModified(CasCorNet):
    def __init__(self, input_size, output_size, args, isotope_masses, gcms_masses):
        super().__init__(input_size, output_size, args)
        self.isotope_masses = isotope_masses
        self.gcms_masses = gcms_masses

    def generate_maccs_mass_string(self):
        maccs_string = self.X_train['MACCS_Fingerprint']
        mass_values = np.round(np.sum(self.X_train, axis=1), 3)
        final_output = maccs_string + '_' + mass_values.astype(str)
        return final_output

    def add_hidden_unit_with_element(self, element):
        if element not in self.isotope_masses:
            raise ValueError("Element not found in isotope masses")

        element_mass = self.isotope_masses[element]
        self.I += 1
        self.X_train = self.augment_input(self.X_train, element_mass)
        self.X_test = self.augment_input(self.X_test, element_mass)

        self.weights = self.init_weights()

        matched = self.check_gcms_match()
        if matched:
            print("Match found in GCMS library")
        else:
            print("No match found, continue adding elements")

    def check_gcms_match(self):
        theoretical_masses = np.sum(self.X_train, axis=1)
        return any(np.isclose(mass, theoretical_masses, atol=0.01) for mass in self.gcms_masses)


def generate_maccs_mass_string():
    return None