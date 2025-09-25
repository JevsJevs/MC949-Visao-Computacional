import kagglehub
import os
import shutil
from pathlib import Path

from canon.utils import image_utils

def download_kaggle_dataset(dataset_slug, project_name):
    """
    Downloads a Kaggle dataset to the specified local path.

    Args:
        dataset_slug (str): The unique identifier for the dataset 
                            (e.g., 'titanic' for 'titanic-dataset' or 
                            'datasnaek/youtube-new').
        path (str): The directory where the dataset should be saved. 
                    Default is the current directory.
    """
    # Create the directory if it doesn't exist
    # path = image_utils.BASE_DATA_PATH / project_name
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #     print(f"Created directory: {path}")

    kaggle_download_path = kagglehub.dataset_download(dataset_slug)
    
    kaggle_download_path = Path(kaggle_download_path)
    
    dest_path = image_utils.BASE_DATA_PATH
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        print(f"Created directory: {dest_path}")
    
    print(kaggle_download_path / project_name)
    print(dest_path)
    
    shutil.move(kaggle_download_path / project_name, dest_path)
    
    # 1
    os.rmdir(kaggle_download_path)
    # versions
    os.rmdir(kaggle_download_path.parent)
    
    
        
        
if __name__ == "__main__":
    datasets = {
        "T1": "eliassantosmartins/mc949-t1",
        "T2":  "eliassantosmartins/mc929-t2"
    }
    
    for project, dataset in datasets.items():
        try:
            download_kaggle_dataset(dataset, project)
        except Exception as e:
            print(e)