# Event-driven taxonomy (EDT) based non-target analysis tool

An Integrated Tool for Text Mining, Deep Learning for Structures Predictions and High-Resolution Mass Spectrometry Applications

## Introduction

The repository is structured into three main folders:

1. **TM**: This module focuses on extracting valuable information from textual data using advanced text mining methodologies.
2. **EDMS**: This module utilizes deep learning to develop methods for identifying ions in HRMS data.
3. **EDT-Screening**: This module is dedicated to building a screening mass spectrometry library for efficient data analysis.

## Features

- **TM**:
  - Data sources query and literature scrapy.
  - Advanced text processing and feature extraction.
  - Keyword analysis and semantic mapping.
  - Integration with external databases for enriched data retrieval.

- **EDMS**:
  - The ANN models for predicting the bioactive substructures (EDFP).
  - The CCNN models for predicting the ions (EDION) based on the EDFPs.
  - Support for custom model training and optimization.
  - The data prepartion and oepn access library decoding.

- **EDT-Screening**:
  - Construction of comprehensive mass spectrometry libraries based on TM and EDMS.
  - Substructures annotation.

## External Links for better TM in our project

- **Web crawler**: [Link](https://github.com/huangjiehui826/sci_hub_crawler_v1/tree/main/sci_hub_crawler_v1)
- **Chemical names recognition from English literature**: [Link](https://github.com/huangjiehui826/chemical_ner_v1)
- **Toxicological terms recognition from English literature**: [Link](https://github.com/huangjiehui826/aop_ner_v1)

## Version Updates
11.10, 2024 Update: Updated the LCMS EDION prediction scripts in the EDMS module.

## Contributing

We welcome contributions from researchers and developers. Please read our Contributing Guidelines to learn how you can contribute.

### Contributors
Special thanks to the following contributors who have made significant contributions to this project:

[Jiehui Huang](https://github.com/huangjiehui826): Contributions in data processing and text mining.
[Liwei He](https://github.com/heliwei30): Contribution in HRMS-based data processing.

Feel free to join this list by contributing to the project!

## License

This project is licensed under the MIT License.

## Acknowledgements

We would like to thank all individuals and organizations that have supported and contributed to this project.
