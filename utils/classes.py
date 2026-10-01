# import pandas as pd
# import math

# # 1) Your 50 target classes (exact matches)
# target_classes = [
#     "airplane flyby",
#     "ambulance siren",
#     "baby crying",
#     "baby laughter",
#     "car engine idling",
#     "cat purring",
#     "chainsawing trees",
#     "church bell ringing",
#     "train whistling",
#     "train horning",
#     "cow lowing",
#     "dog barking",
#     "elk bugling",
#     "fire truck siren",
#     "gibbon howling",
#     "hail",
#     "lawn mowing",
#     "volcano explosion",
#     "stream burbling",
#     "waterfall burbling",
#     "orchestra",
#     "owl hooting",
#     "people cheering",
#     "people crowd",
#     "people marching",
#     "playing bassoon",
#     "playing drum kit",
#     "playing harp",
#     "playing harpsichord",
#     "playing snare drum",
#     "playing timbales",
#     "printer printing",
#     "scuba diving",
#     "sea waves",
#     "sheep bleating",
#     "singing choir",
#     "skiing",
#     "slot machine",
#     "snake hissing",
#     "tractor digging",
#     "underwater bubbling",
#     "railroad car, train wagon",
#     "train wheels squealing",
#     "baltimore oriole calling",
#     "car engine knocking",
#     "driving snowmobile",
#     "wood thrush calling",
#     "hedge trimmer running",
#     "ice cream truck, ice cream van",
#     "woodpecker pecking tree"
# ]

# # Utility to load & clean a CSV of (filename,label)
# def load_df(path):
#     df = pd.read_csv(
#         path,
#         header=None,
#         names=["filename","label"],
#         quotechar='"',
#         skipinitialspace=True
#     )
#     df["label"] = df["label"].str.strip()
#     return df


# # 2) Load your source CSVs
# df_train_src = pd.read_csv("anurag/Cycle_Sound/train.csv", header=None, names=["filename","label"])
# df_test_src  = pd.read_csv("anurag/Cycle_Sound/test.csv",  header=None, names=["filename","label"])

# # 3) Filter down to only our 50 classes
# df_train = df_train_src[df_train_src["label"].isin(target_classes)]
# df_test  = df_test_src [df_test_src ["label"].isin(target_classes)]

# # 4) Now split train → train/val with the new rules:
# train_splits = []
# val_splits   = []

# for cls, group in df_train.groupby("label", sort=False):
#     group = group.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
#     n = len(group)
#     if n > 900:
#         n_train = 850
#         n_val   = 50
#     else:
#         n_train = math.floor(0.9 * n)
#         n_val   = n - n_train

#     train_splits.append(group.iloc[:n_train])
#     val_splits.append( group.iloc[n_train:n_train + n_val] )

# # concatenate and reset index
# train_final = pd.concat(train_splits).reset_index(drop=True)
# val_final   = pd.concat(val_splits)  .reset_index(drop=True)

# # 5) For test, you can keep the same 50‑per‑class rule, or adjust as needed
# test_splits = []
# for cls, group in df_test.groupby("label", sort=False):
#     group = group.sample(frac=1, random_state=42).reset_index(drop=True)
#     test_splits.append(group.iloc[:min(50, len(group))])

# test_final = pd.concat(test_splits).reset_index(drop=True)

# # 6) Write out
# train_final.to_csv("anurag/train_final.csv", header=False, index=False)
# val_final  .to_csv("anurag/val_final.csv",   header=False, index=False)
# test_final .to_csv("anurag/test_final.csv",  header=False, index=False)

# print(f"→ train: {len(train_final)} examples")
# print(f"→   val: {len(val_final)} examples")
# print(f"→  test: {len(test_final)} examples")




import pandas as pd

filename_source = "MACS/download/AVVP_dataset_full.csv"

df = pd.read_csv(filename_source, header=0, sep='\t')
classes = set()

for i in range(len(df)):
    row = df.loc[i, :]
    labels = row.iloc[1]
    labels = labels.split(",")
    for label in labels:
        classes.add(label.strip())

classes = sorted(list(classes))
print(f"Found {len(classes)} unique classes:")
print(classes)