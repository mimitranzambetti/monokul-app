import ddr

model, num_features, indexndex2word_set = ddr.load_model(model_path = 'GoogleNews-vectors-negative300.bin')
print("one")
dicTerms = ddr.terms_from_csv(input_path = 'workplaceDict.csv', delimiter = ',')
print("two")
agg_dic_vecs = ddr.dic_vecs(dic_terms = dicTerms, model = model, num_features = num_features, model_word_set = indexndex2word_set)
