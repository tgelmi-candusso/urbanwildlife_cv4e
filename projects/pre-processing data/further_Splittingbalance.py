
#%% continuation code to non-ran_splitting
### group by categories ##IM HERE IN PROGRESS
    im_dic_df = pandas.DataFrame.from_dict(non_red_dic, orient='index')
    im_dic_df.reset_index()

    for_splitting['group'] = for_splitting.groupby(['0', '2'], sort=False).ngroup() + 1
    
    im_dic_cat = {}
    categs = []
    for item in newdict:
        split_category = newdict[item][0] + "." + newdict[item][2]
        categs.append(split_category)
        im_dic_cat[item] = split_category
    categs = set(categs)
    categs_list = list(categ)
    
    n_cat1={}
    for i in categs_list
        n_cat = {k: im_dic_cat[k] for k in i}
        n_cat1[i] = n_cat
    ###create lists of vlaidation and training datasets (right now is random)
# %%
    for sp in os.listdir(training_folder):
        directory = os.path.join(training_folder, sp)
        if not os.path.isdir(directory):
            continue
        
        ## 
        files_inside_s = os.listdir(directory)
        random.shuffle(files_inside_s)

        if num_images_max is not None or num_images_max > len(files_inside_s):
            # perform random subsampling
            files_inside_s = files_inside_s[0:num_images_max]

        for f in range(len(files_inside_s)):
            files_inside_s[f] = os.path.join(sp, files_inside_s[f])

        train = int(len(files_inside_s)*sample_percent[0])
        val = int(len(files_inside_s)*sample_percent[1])
        # test = int(len(files_inside_s)*sample_percent[2])

        train_samples = files_inside_s[0:train]
        val_samples = files_inside_s[train+1:train+val]
        test_samples = files_inside_s[train+val+1:]     # all the rest for test

        #for sample in train_samples:
        #    write_train.write(sample + '\n')
        #for sample in val_samples:
        #    write_val.write(sample + '\n')
        #for sample in test_samples:
        #    write_test.write(sample + '\n')

    #write_train.close()
    #write_val.close()
    #write_test.close()
