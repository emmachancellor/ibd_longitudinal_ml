import pandas as pd
import numpy as np
import seaborn as sns
import os
import dcor
import math

### Helper Functions ###
def minmax_scale(x, min, max):
    ''' 
    Min/Max Scaler [0,1] that also handles when the max=min.
    Inputs:
    x (int) - Value to be normalized
    min (int) - minmum value
    max (int) - maximum value

    Outputs:
    norm (int) - normalized value [0,1]
    '''
    if min != max:
        scaled = max - min
    else:
        scaled = 1
    norm = (x - min) / scaled
    return norm


def get_mean_val_dict(df, lab_value_names):
    ''' 
    Creates a dictionary with the mean lab values. 

    Inputs:
    df (DataFrame): Contains lab values as columns.
    lab_value_names (list): list of strings that correspond to 
    column headers of each lab value in the DataFrame

    Outputs:
    means_dict (Dict): Dictionary with keys as column headers (str)
    and values as mean lab values (int) 
    '''
    mean_norm_values = []
    for lab in lab_value_names:
        values = df[lab].dropna()
        min_lab = min(values)
        max_lab = max(values)
        mean_value = values.mean()
        norm_lab_val = minmax_scale(mean_value, min_lab, max_lab)
        mean_norm_values.append(norm_lab_val)
    means_dict = dict(map(lambda i,j : (i,j) , lab_value_names, mean_norm_values))
    return means_dict


def whole_population_normalize_and_split(df, folder_path, lab_value_names):
    '''  
    Min-max normalize values column wise in a given DataFrame and 
    separates rows into individual DataFrames grouped by a
    given patient ID. The individual DataFrame contains all rows 
    from the original DataFrame for a given patient ID.

    Inputs:
    df (DataFrame): Contains the values to be normalized. Must
    also include an identifier linking any rows to the same individual.
    file_path
    folder_path (str): Path to a folder to save individual DataFrames
    saved as .csv files
    lab_value_names (list): list of strings that correspond to 
    column headers of each lab value in the DataFrame

    Outputs:
    Returns nothing, but saves individual DataFrames as .csv
    files in the specified folder given by the folder_path
    parameter.
    '''
    mean_norm_values = get_mean_val_dict(df, lab_value_names)
    na_reference = df.copy().isnull()
    for i, lab in enumerate(lab_value_names):
        values = df[lab].dropna()
        min_lab = min(values)
        max_lab = max(values)
        for row in range(len(df)):
            check_missing = na_reference.loc[row, lab]
            if check_missing == False:
                value = df.loc[row, lab]
                norm_val = minmax_scale(value, min_lab, max_lab)
                df.loc[row, lab] = norm_val
    patient_id_lst = list(df.patient_id.unique())
    for patient in patient_id_lst:
        patient_df = df[df['patient_id'] == patient].reset_index(drop=True)
        path = folder_path + str(patient) + '.csv'
        patient_df.to_csv(path, index=False)
    return print("Individual DataFrames saved to: ", folder_path)


def indivdual_normalize_and_split(df, folder_path, lab_value_names):
    '''  
    Min-max normalize values column wise in a given DataFrame and 
    separates rows into individual DataFrames grouped by a
    given patient ID. The individual DataFrame contains all rows 
    from the original DataFrame for a given patient ID.

    Inputs:
    df (DataFrame): Contains the values to be normalized. Must
    also include an identifier linking any rows to the same individual.
    file_path
    folder_path (str): Path to a folder to save individual DataFrames
    saved as .csv files
    lab_value_names (list): list of strings that correspond to 
    column headers of each lab value in the DataFrame

    Outputs:
    Returns nothing, but saves individual DataFrames as .csv
    files in the specified folder given by the folder_path
    parameter.
    '''
    mean_norm_values = get_mean_val_dict(df, lab_value_names)
    patient_id_lst = list(df.patient_id.unique())
    for patient in patient_id_lst:
        patient_df = df[df['patient_id'] == patient].reset_index(drop=True)
        na_reference_df = patient_df.copy()
        na_reference_df = na_reference_df.isnull()
        for i, lab in enumerate(lab_value_names):
            total_missing = patient_df[lab].isnull().sum()
            if total_missing == 5: 
                continue
            else:
                values = patient_df[lab].dropna()
                min_lab = min(values)
                max_lab = max(values)
                for v in range(0, patient_df.shape[0]):
                    is_missing = na_reference_df.loc[v, lab]
                    if is_missing == False:
                        value = patient_df.loc[v, lab]
                        norm_val = minmax_scale(value, min_lab, max_lab)
                        patient_df.loc[v, lab] = norm_val
        path = folder_path + str(patient) + '.csv'
        patient_df.to_csv(path, index=False)
    return print("Individual DataFrames saved to: ", folder_path)


def get_correlation_dict(df, lab_value_names):
    '''  
    Create a dictionary of dictionaries, where the first key is a lab name,
    with the value of a second dictionary. The second dictionary key is the
    name of the lab value between which the correlation distance is calculated
    between the lab named as the key to the first dictionary. The value of the 
    second dictionary is the correlation distance between the two labs.
    Example: {'crp': {'wbc':0.64}}

    Inputs:
    df (DataFrame): Contains lab values in the columns and patients in the rows
    lab_value_names (list): List of strings that represent the column headers
    for each lab value that should be used to calculate a correlation distance.

    Outputs:
    correlation_dict (Dict): Contains distance correlation in the format described
    above. 
    '''
    individual_labs = []
    for lab in lab_value_names: 
        lab_data = df[lab].dropna()
        lab_data = lab_data.to_numpy()
        individual_labs.append(lab_data)
    whole_correlation_dict = dict()
    for i, lab_1 in enumerate(individual_labs):
        all_cors = []
        len_l1 = len(lab_1)
        sub_correlation_dict = dict()
        for j, lab_2 in enumerate(individual_labs):
            len_l2 = len(lab_2)
            min_len = min(len_l1, len_l2)
            red_1 = lab_1[:min_len]
            red_2 = lab_2[:min_len]
            correlation = dcor.distance_correlation(red_1, red_2)
            sub_correlation_dict[lab_value_names[j]] = correlation
        whole_correlation_dict[lab_value_names[i]] = sub_correlation_dict
    return whole_correlation_dict


def get_correlation_matrix(df, lab_value_names):
    '''  
    Create a correlation matrix as a DataFrame describing the distance
    correlation between lab values.

    Inputs:
    df (DataFrame): Contains lab values in the columns and patients in the rows
    lab_value_names (list): List of strings that represent the column headers
    for each lab value that should be used to calculate a correlation distance.

    Outputs:
    correlation_matrix (DataFrame): column headers are the same as index 
    values, thereby making a confusion matrix of distance correlations
    among the lab values. 
    '''
    individual_labs = []
    for lab in lab_value_names: 
        lab_data = df[lab].dropna()
        lab_data = lab_data.to_numpy()
        individual_labs.append(lab_data)
    l_whole_correlation_dict = dict()
    for i, lab_1 in enumerate(individual_labs):
        all_cors = []
        len_l1 = len(lab_1)
        cor_lst = []
        for j, lab_2 in enumerate(individual_labs):
            len_l2 = len(lab_2)
            min_len = min(len_l1, len_l2)
            red_1 = lab_1[:min_len]
            red_2 = lab_2[:min_len]
            correlation = dcor.distance_correlation(red_1, red_2)
            cor_lst.append(correlation)
        l_whole_correlation_dict[lab_value_names[i]] = cor_lst
    correlation_matrix = pd.DataFrame()
    correlation_matrix['lab_name'] = lab_value_names
    correlation_matrix = correlation_matrix.set_index('lab_name')

    for labs, cor_vals in l_whole_correlation_dict.items():
        correlation_matrix[labs] = cor_vals
    return correlation_matrix


def find_neighbors(patient_matrix, visit_v_idx, visit_u_idx, lab_to_impute, correlation_dict):
    ''' 
    Find the distance correlation sum between two given visits. This function
    compares the lab values that are mutually present between two visits, accesses
    the distance correlation between those labs, and sums the distance correlation
    between the lab value that is missing and the lab values that contain measurments
    in both visits. 

    Example: Visit 1 has lab values for cal, pro, fol, fer, hgb, hct, and wbc. Visit 2 has
    values for alt, pro, alb, hgb, pmn, wbc, fer, btwelve, and hct. The value being imputed is 
    plt. This function identifies that both visits have lab values for pro, wbc, fer, hct, and
    hgb. Then, it identifies the distance correlation between pro-plt, wbc-plt, fer-plt, hct-plt,
    and hgb-plt. Finally, the identified distance correlations are summed together, which is then
    used to calculate the distance metric. The inverse distance metric and the 
    value of the comparator visit (visit_u) are saved and used for the output.

    Inputs:
    patient_matrix (DataFrame): contains labs as columns and visits as rows
    visit_d_idx (int): index for the visit that contains the value to be imputed.
    visit_u_idx (int): index for the visit being compared to the visit with the value
    that is to be imputed. 
    lab_to_impute (str): represents the lab type (column header) of the value being imputed

    Outputs:
    weights, values (tuple [list, list]): Weights of each neighbor and the value
    of the neighbor, each stored in a list where the index of the list 
    corresponds to matching weights and values. 
    '''
    visit_v = patient_matrix.loc[visit_v_idx,:].dropna()
    visit_u = patient_matrix.loc[visit_u_idx, :].dropna()
    v_labs = list(visit_v.index)
    u_labs = list(visit_u.index)
    shared_labs = [l for l in v_labs and u_labs if l in v_labs and l in u_labs]
    if len(shared_labs) == 0:
        return []
    correlation_vals = []
    for l in shared_labs:
        val = correlation_dict[lab_to_impute][l]
        correlation_vals.append(val)
    sum_dis_cors = sum(correlation_vals)
    weights_values = []
    for l in shared_labs:
        v_j = visit_v.loc[l]
        u_j = visit_u.loc[l]
        distance_metric = math.sqrt(sum_dis_cors * (v_j - u_j)**2) / sum_dis_cors
        # Calculate the inverse of the distance metric because the smaller the distance
        # the larger the weight (i.e. closer neighbor), we can then say that larger 
        # weights are closer, instead of working with smaller distances.
        weight = 1 / (distance_metric + 1e-6) # add epsilon to avoid division by 0
        weights_values.append((weight, u_j))
    return weights_values


def impute_wknn(k, patient_matrix, mask_matrix, lab, visit_idx, correlation_matrix, correlation_dict, means_dict):
    ''' 
    Impute missing lab values given a patient matrix.

    Inputs:
    k (integer): k-number of neighbors to use when imputing
    patient_matrix (DataFrame): contains labs as columns and visits as rows
    mask_matrix (DataFrame): Boolean mask of the patient matrix where True = null
    and False = int. 
    lab (str): lab type (column header) of the missing value
    visit_idx (int): index in the patient matrix of the visit from which the missing value arises

    Output:
    imputed_value (int): Imputed value for the missing values of a given lab
    for a given patient.
    '''
    # Loop and get data pairs to calculate d(u,v)
    all_weights_values = []
    value = mask_matrix.loc[visit_idx, lab]
    # impute missing value
    for i in range(0, patient_matrix.shape[0]):
        if visit_idx == i: # Do not want to compare a visit to itself
            continue
        if mask_matrix.loc[i, lab] == True: # Make sure the comparing visit has a value
            continue
        else:
            weights_values = find_neighbors(patient_matrix, visit_idx, i, lab, correlation_matrix)
        all_weights_values = all_weights_values + weights_values
    if len(all_weights_values) == 0: 
        # Set imputed value to mean if there are no neighbors
        imputed_value = means_dict[lab]
    else:     
        sorted_weights_vals = sorted(all_weights_values)
        neighbors = sorted_weights_vals[:k]  
        w_knn, v_knn = map(list,zip(*sorted_weights_vals))
        v_knn = np.asarray(v_knn)
        w_knn = np.asarray(w_knn)
        norm_weights = w_knn / sum(w_knn)
        imputed_value = sum(v_knn * norm_weights)
    return imputed_value

def combine_dataframes(imputed_folder_path, imputed_file_path):
    ''' 
    Combines individual DataFrames into one DataFrame. Note: this
    function assumes that the DataFrame contains specific column
    headers. Please see README for more details.

    Inputs:
    imputed_folder_path (str): Folder path to where imputed individual
    files are saved.
    imputed_file_path (str): Path to save single .csv file of combined DataFrames

    Outputs:
    Returns nothing. Saves combined DataFrame at the location specified by
    the imputed_file_path parameter.
    '''
    impute_file_lst = os.listdir(imputed_folder_path) 
    column_labels = []
    visit_lst = []
    all_patients_df = pd.DataFrame()
    for pt in impute_file_lst:
        if '.D' in pt:
            continue
        path = imputed_folder_path + '/' + pt 
        imputed_pt = pd.read_csv(path)
        imputed_pt = imputed_pt.drop('year', axis=1)
        for i in range(0,imputed_pt.shape[0]):
            label = str(i + 1)
            visit = imputed_pt.iloc[i, 1:]
            patient_id = imputed_pt.loc[0, 'patient_id']
            column_labels = column_labels + [x + label for x in list(visit.index)]
            visit_lst = visit_lst + visit.to_list() 
        full_dict = dict(zip(column_labels, visit_lst))
        df = pd.DataFrame(full_dict, index=[0])
        df['patient_id'] = patient_id
        all_patients_df = pd.concat([all_patients_df, df], axis=0 )
    all_patients_df.to_csv(imputed_file_path, index=False)
    return print("Imputed DataFrames have been combined and saved as: ", imputed_file_path)

### Main Function ###
def dc_wknn_imputer(df, folder_path, imputed_folder_path, imputed_file_path, level='individual'):
    ''' 
    Performs distance correlation weighted KNN imputation
    on a given dataset. Note: the dataset must contain 
    specific column headers. See README for exact DataFrame format
    specifications for compatability with this function.

    Inputs:
    df (DataFrame): DataFrame to perform imputation on. 
    folder_path (str): Folder path where NON-IMPUTED individual 
    files can be saved after splitting the larger dataframe into
    patient-level dataframes.
    imputed_file_path (str): Path to save single .csv file of combined DataFrames
    level (str): Denotes which normalize and split method to use. 'individual' 
    will split the dataframe and normalize patients at the patient level.
    'whole_pop' will normalize values across the whole dataframe before splitting at the patient level.

    Outputs: 
    Returns nothing. Imputed individual DataFrames are saved in the 
    folder specified by impute_folder_path.
    '''
    lab_value_names = list(df.columns.values)
    lab_value_names = lab_value_names[1:-1]
    if level == 'individual':
        indivdual_normalize_and_split(df, folder_path, lab_value_names)
    if level == 'whole_pop':
        whole_population_normalize_and_split(df, folder_path, lab_value_names)
    correlation_dict = get_correlation_dict(df, lab_value_names)
    correlation_matrix = get_correlation_matrix(df, lab_value_names)
    means_dict = get_mean_val_dict(df, lab_value_names)
    split_file_lst = os.listdir(folder_path)
    for pt in split_file_lst: 
        if '.D' in pt:
            continue
        path = folder_path + '/' + pt 
        patient_matrix = pd.read_csv(path)
        patient_id = patient_matrix.loc[0, 'patient_id']
        patient_matrix = patient_matrix.reset_index(drop=True)
        # Create a dataframe containing only the lab values:
        labs_only_patient_matrix = patient_matrix.drop(['patient_id', 'year'], axis=1)
        mask_visits = labs_only_patient_matrix.isnull()
        for lab in lab_value_names: # look through each lab in a given visit
            for n in range(0, labs_only_patient_matrix.shape[0]): 
                value = mask_visits.loc[n, lab] # check the value of the lab
                if value == True: # value is missing — impute
                    imputed_value = impute_wknn(3, 
                                                labs_only_patient_matrix, 
                                                mask_visits, 
                                                lab, 
                                                n, 
                                                correlation_matrix,
                                                correlation_dict,
                                                means_dict)
                    patient_matrix.loc[n, lab] = imputed_value
        patient_path = imputed_folder_path + '/'+ str(patient_id) + '_imputed.csv'
        patient_matrix.to_csv(patient_path, index=False)
    combine_dataframes(imputed_folder_path, imputed_file_path)
    return ("Imputation Complete")


### Run Function on Patient Datasets ###
# Load Datasets (MacBook Pro)
uc = pd.read_csv('/Users/emmadyer/Desktop/long_ibd_data/data/expanded_raw/v2_uc_expanded.csv')
cd = pd.read_csv('/Users/emmadyer/Desktop/long_ibd_data/data/expanded_raw/v2_cd_expanded.csv')
healthy = pd.read_csv('/Users/emmadyer/Desktop/long_ibd_data/data/expanded_raw/v2_healthy_expanded.csv')

dfs = [uc, cd, healthy]

split_folders = [ '/Users/emmadyer/Desktop/long_ibd_data/data/uc_pts/',
                '/Users/emmadyer/Desktop/long_ibd_data/data/cd_pts/',
                '/Users/emmadyer/Desktop/long_ibd_data/data/healthy_pts/']

imp_file_paths = ['/Users/emmadyer/Desktop/long_ibd_data/data/v3_uc_imputed.csv',
                    '/Users/emmadyer/Desktop/long_ibd_data/data/v3_cd_imputed.csv',
                    '/Users/emmadyer/Desktop/long_ibd_data/data/v3_healthy_imputed.csv']

imp_folder_paths = [ '/Users/emmadyer/Desktop/long_ibd_data/data/uc_imputed',
                    '/Users/emmadyer/Desktop/long_ibd_data/data/cd_imputed',
                    '/Users/emmadyer/Desktop/long_ibd_data/data/healthy_imputed']

for i, df in enumerate(dfs):
    dc_wknn_imputer(df, split_folders[i], imp_folder_paths[i], imp_file_paths[i])

disease_codes = [2,1,0]
all_imp = []

for i, f in enumerate(imp_file_paths):
    df = pd.read_csv(f)
    df['ibd_disease_code'] = disease_codes[i]
    all_imp.append(df)

all_imputed = pd.concat(all_imp, axis=0)
all_imputed.to_csv('/Users/emmadyer/Desktop/long_ibd_data/data/v3_all_labs_imputed.csv', index = False)
