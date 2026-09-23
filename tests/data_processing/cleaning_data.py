from src.data_loader.fetch import load_raw_data
from src.data_processing.explore import print_basic_stats
from src.data_processing.clean import clean_data

# first testr
print_basic_stats(load_raw_data())
print_basic_stats(clean_data(load_raw_data()))