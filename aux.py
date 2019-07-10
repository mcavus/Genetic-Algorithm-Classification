'''
CS 550: Machine Learning
Homework 3 - Auxiliary functions 
Muhammed Cavusoglu, 21400653
'''

def get_selected_features(selected_features, all_features):
    selected_f = []
    for sample in all_features:
        s_f = []
        for i in range(len(selected_features)):
            if selected_features[i] == 1:
                s_f.append(sample[i])
        selected_f.append(s_f)
    return selected_f

def get_predicted_labels(class_prob):
    predicted_labels = []
    
    for p in class_prob:
        predicted_labels.append(p.argmax() + 1)
            
    return predicted_labels
    
def get_class_miss_percentages(true_labels, predicted_labels):
    c1_missed = 0
    c1_tot = 0
    c2_missed = 0
    c2_tot = 0
    c3_missed = 0
    c3_tot = 0
    
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            c1_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c1_missed += 1
        
        if true_labels[i] == 2:
            c2_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c2_missed += 1
                
        if true_labels[i] == 3:
            c3_tot += 1
            if true_labels[i] != predicted_labels[i]:
                c3_missed += 1
        
    c1_miss_percent = (100.00 * c1_missed) / c1_tot  
    c2_miss_percent = (100.00 * c2_missed) / c2_tot
    c3_miss_percent = (100.00 * c3_missed) / c3_tot
    
    if c1_miss_percent <= 1:
        c1_miss_percent = 1
    if c2_miss_percent <= 1:
        c2_miss_percent = 1
    if c3_miss_percent <= 1:
        c3_miss_percent = 1
        
    print "Missed samples for each class: ", c1_missed, c2_missed, c3_missed
    
    return c1_miss_percent, c2_miss_percent, c3_miss_percent
    
def load_data():
    train = open("ann-train.data", "r")
    training_data = []
    for line in train:
        training_data.append(line.strip().split(" "))
    
    convert_types(training_data)
    
    test = open("ann-test.data", "r")
    test_data = []
    for line in test:
        test_data.append(line.strip().split(" "))
    
    convert_types(test_data)
    
    training_features = []
    training_labels = []
    for sample in training_data:
        training_features.append(sample[:-1])
        training_labels.append(sample[-1])
        
    test_features = []
    test_labels = []
    for sample in test_data:
        test_features.append(sample[:-1])
        test_labels.append(sample[-1])
        
    cost = open("ann-thyroid.cost", "r")
    costs = []
    for line in cost:
        costs.append(float(line.strip().split(":")[1]))
    costs.append(0) # 21st feature is a comb. of 19th and 20th features
        
    return training_features, training_labels, test_features, test_labels, costs
    
def convert_types(data):
    for row in data:
        row[0] = float(row[0].strip())
        row[1] = int(row[1].strip())
        row[2] = int(row[2].strip())
        row[3] = int(row[3].strip())
        row[4] = int(row[4].strip())
        row[5] = int(row[5].strip())
        row[6] = int(row[6].strip())
        row[7] = int(row[7].strip())
        row[8] = int(row[8].strip())
        row[9] = int(row[9].strip())
        row[10] = int(row[10].strip())
        row[11] = int(row[11].strip())
        row[12] = int(row[12].strip())
        row[13] = int(row[13].strip())
        row[14] = int(row[14].strip())
        row[15] = int(row[15].strip())
        row[16] = float(row[16].strip())
        row[17] = float(row[17].strip())
        row[18] = float(row[18].strip())
        row[19] = float(row[19].strip())
        row[20] = float(row[20].strip())
        row[21] = int(row[21].strip())