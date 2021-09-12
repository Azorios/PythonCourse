# PythonCourse

# How to junyper notebooks 
https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html#get-started

# K nearest neighbor:
    - 2 vectors x, y => axis
    - we have training samples and for each new sample we calculate the euclidian distance to every training sample
    - find the k nearest neighbors => group
    - choose/predict the label of this group based on the most common class labels,
        e.g. k=3 blue=2 green=1 => label=blue

#TODO
    - euclidian distance function:
        -of 2 points: d = sqrt((x2-x1)**2 + (y2-y1)**2)
        -general case: look at formula => x1,x2 : np.sqrt(np.sum((x1-x2)**2))
    - generate training sample
    - plot training sample
    - compute distances from new sample to training samples
    - get k nearest samples and labels
    - majority vote: get the most common label

    -accuracy of predictions

    -special cases: equal number of labels => which category?