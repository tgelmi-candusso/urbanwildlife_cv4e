filename = '/home/mgonzalez3/urbanwildlife_cv4e_max/projects/pre-processing data/categories.csv'
# count = -2

# with open(filename) as file:
#     for line in file:
#         count += 1
#         line = str(count) + ',' + line 
#         print(line,  end="")

# count = 0
# arr = []
# for i in range(62):
#     arr.append(999)
# print(len(arr))
# print(arr)

speciesArr = []
with open(filename) as file:
    for line in file:
        new_str = ','.join(line.split(',')[1:])
        speciesArr.append(new_str.rstrip())
print(len(speciesArr))
