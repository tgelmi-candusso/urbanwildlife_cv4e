### in case a wrong file or subfolder has been included in the TVT files. Splitting had to be ran again though.
test = r'/datadrive/animals_training_dataset/tuw_uwin_nonred/split_random/test.txt'
test_fixed = r'/datadrive/animals_training_dataset/tuw_uwin_nonred/split_random/test_f.txt'

import os

with open(test, "r") as input:
    with open(test_fixed, "w") as output:
        # iterate all lines from file
        for line in input:
            # if text matches then don't write it
            if line.strip("\n") != 'deer/White-tailed deer':
                output.write(line)

# replace file with original name
os.replace(test, test_fixed)


with open(test) as f:
    lines = f.readlines()

print([s for s in lines if s.endswith('r\n')])

