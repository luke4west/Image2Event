import pandas as pd 
from sklearn.model_selection import train_test_split

dataset_size = 7000
img_df = pd.read_csv("./data/LateOrchestration/Event_Domain_Adaptation/labeled_data.csv")
img_df = img_df.sample(n=dataset_size, random_state=0, replace=True).reset_index(drop=True)
print(len(img_df))

event_df = pd.read_csv("./data/LateOrchestration/Event_Domain_Adaptation/event_data.csv")
event_df = event_df.sample(n=dataset_size, random_state=0, replace=True).reset_index(drop=True)

df = pd.DataFrame()
df["image"] = img_df["image"]
df["event"] = event_df["event"]
df.to_csv("./data/LateOrchestration/Event_Domain_Adaptation/frame2event_dataset.csv", index=False)

train_, val_test = train_test_split(df, test_size=0.5, random_state=0)
val_, test_ = train_test_split(val_test, test_size=0.5, random_state=0)

train_.to_csv("./data/LateOrchestration/Event_Domain_Adaptation/frame2event_train_dataset.csv", index=False)
val_.to_csv("./data/LateOrchestration/Event_Domain_Adaptation/frame2event_val_dataset.csv", index=False)
test_.to_csv("./data/LateOrchestration/Event_Domain_Adaptation/frame2event_test_dataset.csv", index=False)