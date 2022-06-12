import pandas as pd
import numpy as np


note_lookup = pd.read_csv("data/midi-notes.csv").iloc[:-22]
note_lookup["MIDI"] = note_lookup["MIDI"].astype(np.int16)
note_lookup["NOTE"] = note_lookup["NOTE"].str.split("/").str[0]
note_lookup = note_lookup.set_index("MIDI")