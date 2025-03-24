import os

train = os.environ.get("SM_CHANNEL_TRAIN")
val = os.environ.get("SM_CHANNEL_VALIDATION")

train_dir = os.listdir(train)
print(f" THIS IS TRAIN {train_dir}")