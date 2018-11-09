import pandas as pd

# Download the subset msd summary file from https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset#subset
# key is the location of the data set inside h5 file
songs_summary = pd.read_hdf('MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5', key="metadata/songs")

print(songs_summary.head())
