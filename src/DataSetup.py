import os
import pickle
import shutil
import tensorflow as tf

###function to set up the folder structure for the data
def folder_setup(image_classes, base_folder = 'data'): 

    #dict to store the paths for the train and test sets
    train_test_path_dict = {
        'train':set(),
        'test':set()
    }

    #base folder for the data
    if not os.path.isdir(base_folder):
      os.mkdir(base_folder)

    #folder names: train/test and class labels
    train_test_folders = ['train', 'test']
    classes = image_classes

    #create the train/test folders
    for folder in train_test_folders:

        folder_path = os.path.join(base_folder, folder)

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        #create a folder for each class within the train and test folders
        for img_class in classes:

            class_folder_path = os.path.join(folder_path, img_class)

            #add the path to the path dict
            train_test_path_dict[folder].add(class_folder_path)

            if not os.path.isdir(class_folder_path):
              os.mkdir(class_folder_path)
  
    return train_test_path_dict


#train_test_split method to ensure the train/test split is consistent in each notebook
def train_test_split(saved_resources_path):

    #get the train/test file lists for each image class 
    with open(os.path.join(saved_resources_path,'filepath_dict.pkl'), 'rb') as file_in:
        filepath_dict = pickle.load(file_in)

    #get the mappings of image class to  original file path 
    with open(os.path.join(saved_resources_path,'dict_original_image_paths.pkl'), 'rb') as file_in:
        dict_original_image_paths = pickle.load(file_in)

    #populate each image class folder for both the train and test sets
    for filepath in filepath_dict.keys():

        #get the image class
        image_class = filepath.split('/')[-1]
        
        #get the file path to the raw data for the image class
        raw_data_path = dict_original_image_paths[image_class]
        
        #get the list of of files to move to the current filepath
        destination_file_list = filepath_dict[filepath]

        #populate the current folder with the appropriate files
        for filename in os.listdir(raw_data_path):
            if filename in destination_file_list:
                source_file = os.path.join(raw_data_path, filename)
                destination_file = os.path.join(filepath, filename)
                shutil.copyfile(source_file, destination_file)



#function to create the train and validation sets from the image directories
def training_datasets_from_directory(random_state, image_size, batch_size):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        "data/train",
        validation_split=0.1,
        subset="training",
        shuffle=True,
        seed=random_state,
        #image_size=(constants_dict['IMAGE_SIZE'][0]//2, constants_dict['IMAGE_SIZE'][1]//2)
        image_size=image_size,
        batch_size=batch_size,
    )

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        "data/train",
        validation_split=0.1,
        subset="validation",
        shuffle=True,
        seed=random_state,
        #image_size=(constants_dict['IMAGE_SIZE'][0]//2, constants_dict['IMAGE_SIZE'][1]//2)
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_data, validation_data


#function to create the test set from the image directories
def test_dataset_from_directory(random_state, image_size, batch_size):
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        "data/test",
        shuffle=False,
        seed=random_state,
        #image_size=(constants_dict['IMAGE_SIZE'][0]//2, constants_dict['IMAGE_SIZE'][1]//2),
        image_size=image_size,
        batch_size=batch_size
    )
    return test_data