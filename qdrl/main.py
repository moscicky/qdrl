from loader import QueryDocumentDataset
from config import load_config, TrainingConfig

if __name__ == '__main__':
    config_path = 'resources/configs/training.yml'
    config = load_config(config_path)

    ds = QueryDocumentDataset(config)



