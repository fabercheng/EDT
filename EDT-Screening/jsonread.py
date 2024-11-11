import json
from collections import defaultdict


# Recursive function to traverse nested dictionaries and lists, counting occurrences of each variable
def count_variables(data, variable_count):
    if isinstance(data, dict):
        for key, value in data.items():
            variable_count[key] += 1
            count_variables(value, variable_count)  # Recursively process nested dictionaries and lists
    elif isinstance(data, list):
        for item in data:
            count_variables(item, variable_count)


# Function to export selected variables to a new JSON file
def export_selected_variables(data, selected_keys, output_filename):
    # Filter records based on selected keys
    filtered_data = []
    for record in data:
        filtered_record = {key: record[key] for key in selected_keys if key in record}
        filtered_data.append(filtered_record)

    # Save the filtered data to a new JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Exported selected variables to {output_filename}")


# Load JSON file ## 'MassBank.json'
with open('MoNA.json' , 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count the number of records
record_count = len(data)
print(f"Number of records: {record_count}")

# Initialize dictionary to count variable occurrences
variable_count = defaultdict(int)

# Count each variable in all records
for record in data:
    count_variables(record, variable_count)

# Print all variables with their counts
print("\nVariable count (including nested variables):")
for variable, count in variable_count.items():
    print(f"{variable}: {count}")

# Prompt the user to select variables for export
selected_keys = input("\nEnter variable names to export, separated by commas: ").split(',')

# Export selected variables to a new JSON file
export_selected_variables(data, selected_keys, 'exported_variables.json')
