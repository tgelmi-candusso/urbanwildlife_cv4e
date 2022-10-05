
######check how many photos per train val test
train = r'/datadrive/animals_training_dataset/tuw_uwin_nonred2/split_random/train.txt'
val = r'/datadrive/animals_training_dataset/tuw_uwin_nonred2/split_random/val.txt'
test = r'/datadrive/animals_training_dataset/tuw_uwin_nonred2/split_random/test.txt'

with open(train) as f:
    train_df = f.readlines()
with open(val) as f:
    val_df = f.readlines()
with open(test) as f:
    test_df = f.readlines()

print(len(train_df))
print(len(val_df))
print(len(test_df))