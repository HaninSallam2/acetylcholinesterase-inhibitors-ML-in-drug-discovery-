# acetylcholinesterase-inhibitors-ML-in-drug-discovery-

This project implements the Tuna Swarm Optimization (TSO) algorithm to find optimal hyperparameters for various machine-learning models. It includes implementations for Random Forest, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN), and Multilayer Perceptron (MLP) classifiers, and a stacking model on down samples and over samples and unbalanced acetylcholinesterase inhibitors targets that was downloaded from Chembel and preprocessed and labeled into active, inactive, and intermediate molecules based on the IC50 values. It also contains feature selection using Variance Thresholding and Recursive Feature Elimination (RFE). 

## Pipeline overview

- Downloaded acetylcholinesterase human targets from Chembel and preprocessed the data by removing duplicates and NAs
- Got the PubChem molecular descriptors from PaDEL and divided th molecules into 3 classes (active, inactive, and intermediate)
- Over-sampling and under-sampling
- Removed low-variance features
- Recursive Feature Elimination (RFE) with random forest
- implemented TSO 
  ![image](https://github.com/user-attachments/assets/5cdd973e-bb2c-4ff3-b64a-8b8a74499ecb)
- Executed the hyperparameters and best accuracies of each model from the tso_optimization() function
### Prerequisites

- Python 3.7 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `math`
- **PaDEL-Descriptor** needs to be downloaded and installed and added to your environment variables or linked through shell script (padel.sh)

### Installation


1.  **Install the required libraries:**

    ```bash
    pip install pandas numpy scikit-learn
    ```

2. **Download and setup PaDEL-Descriptor:**

  -  Download PaDEL-Descriptor with 
! wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip
! wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.sh
## Usage
### Data Preparation

  **Prepare your PaDEL shell script**
      - The `padel.sh` script should contain the correct path to `PaDEL-Descriptor` . Ensure it is set up to read from `molecule.smi` and output a CSV file named `descriptors_output.csv`. Make sure that  `padel.sh` and `molecule.smi` files are in the same directory as your main script.


### Running the Script
1.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    - The script will perform the following steps:
      - Process the bioactivity data by converting standard values to pIC50 and classifying activity labels.
      - Generates PubChem fingerprints using PaDEL-Descriptor.
      - Merges generated descriptors with bioactivity data and saves the generated data into .csv files.
      - Performs feature selection and hyperparameter optimization using the TSO algorithm for the Random Forest model.
      - Saves the selected features to .csv files.
    - The script will output the best hyperparameter values and best cross-validation accuracy achieved during the optimization process for the Random Forest.

## Implemented Algorithms

### Random Forest (RF)

-   `fitness_function_rf`: Evaluates Random Forest model performance by tuning `n_estimators`, `min_samples_split`, and `min_samples_leaf`.
-   `search_space_rf`: Defines the search range for RF hyperparameters.

### Support Vector Machine (SVM)

-   `fitness_function_svm`: Evaluates SVM model performance by tuning `C` and `gamma`.
-   `search_space_svm`: Defines the search range for SVM hyperparameters.

### Decision Tree (DT)

-   `fitness_function_dt`: Evaluates Decision Tree model performance by tuning `max_depth`, `min_samples_split`, and `min_samples_leaf`.
-   `search_space_dt`: Defines the search range for DT hyperparameters.

### K-Nearest Neighbors (KNN)

-   `fitness_function_knn`: Evaluates KNN model performance by tuning `n_neighbors` and `weights`.
-   `search_space_knn`: Defines the search range for KNN hyperparameters.

### Multilayer Perceptron (MLP)

-   `fitness_function_mlp`: Evaluates MLP model performance by tuning `layer_1_size`, `layer_2_size`, `dropout_rate`, `learning_rate` and `batch_size`.
-   `search_space_mlp`: Defines the search range for MLP hyperparameters.

### Stacking

- `fitness_function_stacking`: Evaluates Stacking Classifier model performance by tuning hyperparameters of the base models.
- `search_space_stacking`: Defines the search range for the stacking hyperparameters.


## Optimization Algorithm

### Tuna Swarm Optimization (TSO)

The project implements a modified version of the Tuna Swarm Optimization algorithm, tailored for hyperparameter tuning. The core logic is as follows:

-   `initialize_population`: Initializes a population of random hyperparameter sets for the algorithm.
-   `evaluate_population`: Evaluates fitness of each hyperparameter configuration (tuna) in the current population.
-   `update_tuna_positions`: Updates each tuna's position by a combination of position update in the swarm and random search in parameter space.
-   `tso_optimization`: Iteratively performs the optimization procedure: evaluation, update, and selection of the current best parameters, stopping when the maximum iteration count is reached or no improvement in best fitness is observed for a certain number of iterations.

## Feature Selection

This project uses:

-   **Variance Thresholding:** To remove features with low variance, filtering out constant and almost-constant variables.
-   **Recursive Feature Elimination (RFE):** A method to select features by recursively considering smaller and smaller sets of features by removing features based on the model's importance.

## Data Preprocessing

### Bioactivity Data Processing

The script performs several preprocessing steps on the bioactivity data:

-   Converts `standard_value` to numeric and removes `NaN` values.
-   Converts `standard_value` of type `pIC50` to `IC50` values in uM.
-   Classifies activity based on standard values:
    - `inactive` for values >= 300
    - `active` for values <= 100
    - `intermediate` for values between 100 and 300.
-   Removes values outside of range for the data
 -   Calculates pIC50 values from IC50 values.
-   Averages pIC50 values for molecules with duplicate `canonical_smiles` and selects the most common class.

### Molecular Descriptor Generation

-   The `padel.sh` script utilizes PaDEL-Descriptor to generate PubChem fingerprints for the molecules from the `molecule.smi` file.

### Data Merging

-   The generated descriptor file from PaDEL, named `descriptors_output.csv` is merged with the preprocessed bioactivity data based on matching molecule IDs to produce final datasets.
-   Two final datasets are created: one with the class labels (`bioactivity_discriptors_3class_pubchem_fp_final.csv`) and the other with the pIC50 values (`bioactivity_discriptors_pic50_pubchem_fp_final.csv`)

## Results and Evaluation

The optimization process will output:

-   The best hyperparameter values found.
-   The corresponding cross-validation accuracy.

The script will also save:

-   The features selected after Variance Thresholding as `X_low_variance_removed_3class.csv`.
-   The selected features after Recursive Feature Elimination as `X_recursive_feature_elimination.csv`.
-   The dataset with  bioactivity labels (`bioactivity_discriptors_3class_pubchem_fp_final.csv`)
-   The dataset with pIC50 values  (`bioactivity_discriptors_pic50_pubchem_fp_final.csv`)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
