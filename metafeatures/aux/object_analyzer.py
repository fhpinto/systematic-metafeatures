"""
Input: a pandas (?) dataframe
Output: a description of the dataframe schema: how many variables it contains of each datatyope

Note: in the future we could extend this function in order to able to deal with more complex objects
"""
#ToDo: ideally, this should take a pandas dataframe or a numpy array as input and output the following list
#for a classification dataset
#{
#    "numericalAttrs": [name of var1], [name of var2],
#    "categoricalAttrs": [name of var3],
#    "classLabel": [name of target column],
#    "regLabel": []
#}