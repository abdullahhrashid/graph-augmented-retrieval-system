from datasets import load_dataset
from src.utils.logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

#setting up logging
logger = get_logger(__name__)

#path for saving splits
path = os.path.join(os.path.dirname(__file__), '../data/raw')

logger.info('Downloading dataset')
dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='train')

logger.info('Filtering out easy queries and selecting 25000 samples for training')
train_set = dataset.filter(lambda x: x['level'] != 'easy').select(range(25000))

logger.info('Selecting 5000 samples for validation')
val_set = dataset.filter(lambda x: x['level'] != 'easy').select(range(25000, 30000))

logger.info('Selecting 5000 samples for testing')
test_set = dataset.filter(lambda x: x['level'] != 'easy').select(range(30000, 35000))

logger.info('Saving datasets')
train_set.save_to_disk(os.path.join(path, 'train'))
val_set.save_to_disk(os.path.join(path, 'val'))
test_set.save_to_disk(os.path.join(path, 'test'))

logger.info('Datasets saved successfully')
