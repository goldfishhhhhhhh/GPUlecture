import  random
def custom_data_split(numfolders, images_path_sep_folders, label_list_sep_folders, dataset_split_percentage):
    datasplit_share = dataset_split_percentage/100 # This define how many percentage of dataset is training and validation split.
    folder_indexes = list(range(numfolders))
    #random.shuffle(folder_indexes)
    training_data_folders = folder_indexes[:int(numfolders * datasplit_share)]
    validation_data_folders = folder_indexes[int(numfolders * datasplit_share):]
    training_paths = []
    training_labels = []
    for i in training_data_folders:
        training_paths.extend(images_path_sep_folders[i])
        training_labels.extend(label_list_sep_folders[i])
    training_path_plus_labels = list(zip(training_paths, training_labels))
    validation_paths = []
    validation_labels = []
    for i in validation_data_folders:
        validation_paths.extend(images_path_sep_folders[i])
        validation_labels.extend(label_list_sep_folders[i])
    validation_path_plus_labels = list(zip(validation_paths, validation_labels))

    return training_path_plus_labels, validation_path_plus_labels
