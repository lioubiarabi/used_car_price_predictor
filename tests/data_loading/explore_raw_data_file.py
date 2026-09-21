from src.data_loader.fetch import load_raw_data
from src.data_processing.explore import print_basic_stats

# first test
print_basic_stats(load_raw_data())