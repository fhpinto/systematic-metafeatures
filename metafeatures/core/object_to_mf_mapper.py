import itertools

def map_object_to_mf(attr_dict, mf):

    tickets = []

    numAttr = {k for k, v in attr_dict.items() if v['type'] == 'numerical'}
    catAttr = {k for k, v in attr_dict.items() if v['type'] == 'categorical'}
    regLabel = {k for k, v in attr_dict.items() if (v['type'] == 'numerical') and ( v['is_target'] == True )}
    classLabel = {k for k, v in attr_dict.items() if (v['type'] == 'categorical') and ( v['is_target'] == True )}

    # MATRIX OBJECTS. Example: count number of rows.
    if mf.get_matrix_applicable() == True:
        raise NotImplementedError

    # METAFUNCTION THAT CAN HANDLE BOTH NUMERICAL AND CATEGORICAL FEATURES
    if mf.get_numerical_arity()==mf.get_categorical_arity():
        raise NotImplementedError

    # CATEGORICAL ARITY = 1. Example: entropy
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(classLabel)!=0):
        tickets += [list(catAttr)[i:i + 1] for i in range(0, len(catAttr))]
        tickets += [list(classLabel)[i:i + 1] for i in range(0, len(classLabel))]
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(regLabel)!=0) and (len(catAttr)==0):
        tickets += []
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(regLabel)!=0) and (len(catAttr)>=1):
        tickets += [list(catAttr)[i:i + 1] for i in range(0, len(catAttr))]

    # CATEGORICAL ARITY = 2. Example: mutual information
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)!=0) and (len(catAttr)>=1):
        tickets += [list(subset) for subset in itertools.combinations(catAttr, 2)]
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)==1) and (len(catAttr)==0):
        tickets += []
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)>=2) and (len(catAttr)==0):
        tickets += [list(classLabel)[i:i + 1] for i in range(0, len(classLabel))]

    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(regLabel)!=0) and (len(catAttr)>=2):
        raise NotImplementedError
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(regLabel)!=0) and (len(catAttr)<=1):
        tickets += []

    # NUMERICAL ARITY = 1. Example: average.
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(regLabel)!=0):
        tickets += [list(numAttr)[i:i + 1] for i in range(0, len(numAttr))]
        tickets += [list(regLabel)[i:i + 1] for i in range(0, len(regLabel))]
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)==0):
        tickets += []
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)>=1):
        tickets += [list(numAttr)[i:i + 1] for i in range(0, len(numAttr))]

    # NUMERICAL ARITY = 2. Example: Pearson's correlation.
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)>=2):
        raise NotImplementedError
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)<=1):
        tickets += []
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)!=0) and (len(numAttr)>=1):
        raise NotImplementedError
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)==1) and (len(numAttr)==0):
        tickets += []
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)>=2) and (len(numAttr)==0):
        tickets += [list(regLabel)[i:i + 1] for i in range(0, len(regLabel))]

    return tickets


