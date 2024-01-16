from pathlib import Path
import os


def path_label_list(data_path):
    all_images_path = []
    all_images_path_sep_folders = []
    data_dir = Path(data_path)
    folders_path = list(data_dir.glob('*'))
    #all_images_path = list(data_dir.glob('*\*.jpg')) #error note: I had high accuracy so I checked the code. the sort of files in this line was not write. my chatgpt quest:I used this code yyy = os.listdir(folder_path). Now in yyy which is a list of names of all images the order how it save all the names has a problem. I want to save all names in the order of their names but apparently it is using something else. Cause the first file sould be output_001.jpg but it is output_1177.jpg
    #so I used manual soarting for files name
    sorted_folders_path = folders_path
    #sorted_folders_path = sorted(folders_path, key=lambda x: int(x.name))
    for temp_folder_path in sorted_folders_path:
        unsorted_images_names = os.listdir(temp_folder_path)
        sorted_images_names = sorted(unsorted_images_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
        temp_path_images = [os.path.join(temp_folder_path, file_name) for file_name in sorted_images_names]
        all_images_path_sep_folders.append(temp_path_images)
        all_images_path.extend(temp_path_images)

    folder_names = [os.path.basename(path) for path in sorted_folders_path]

    label_mapping = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6
    }

    def create_label_list(file_path, label_mapping):
        with open(file_path, 'r') as file:
            file_content = file.readlines()
        label_list = [label_mapping[line.split('\t')[1].strip()] for line in file_content[1:]]
        return label_list

    combined_label_list = []
    combined_label_list_sep_folders = []
    phase_labels_file_path_list = list(Path('E:\\Ali\\December2023\\phase_annotations\\').glob('*'))
    for number_label_file in range(len(folders_path)):
        label_file_path = phase_labels_file_path_list[number_label_file]
        label_list = create_label_list(label_file_path, label_mapping)
        label_list = label_list[3::25]
        combined_label_list.extend(label_list)
        combined_label_list_sep_folders.append(label_list)

    merged_list = list(zip(all_images_path, combined_label_list))
    return merged_list, all_images_path_sep_folders, combined_label_list_sep_folders



