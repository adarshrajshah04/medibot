import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv(r"C:\Users\SNEHA\OneDrive\Desktop\New folder\Training.csv")
testing = pd.read_csv(r"C:\Users\SNEHA\OneDrive\Desktop\New folder\Testing.csv")
cols = training.columns[:-1]
x = training[cols].copy()
y = training['prognosis']
y1 = y

# Encode string features in x and testx
for col in x.columns:
    if x[col].dtype == 'object':
        le_x = preprocessing.LabelEncoder()
        x[col] = le_x.fit_transform(x[col])
        if col in testing:
            testing[col] = le_x.transform(testing[col])

reduced_data = training.groupby(training['prognosis']).max()

#mapping string to nmbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 42)

testx = testing[cols].copy()
testy = testing['prognosis']
for col in testx.columns:
    if testx[col].dtype == 'object':
        le_tx = preprocessing.LabelEncoder()
        testx[col] = le_tx.fit_transform(testx[col])

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

print(clf.score(x_train, y_train))
print("cross result=========")

scores = cross_val_score(clf, x_test, y_test, cv=3)

print(scores)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 30)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

severityDictionary = dict()
dedcription_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptoms in enumerate(x):
    symptoms_dict[symptoms] = index
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary.get(item, 0)
    if ((sum*days)/(len(exp)+1)>13):
        print("You should take the consultaiton from doctor")
    else:
        print("It might not be that bad you should take precaution.")

def getDescription():
    global dedcription_list
    with open('C:\\Users\\SNEHA\\OneDrive\\Desktop\\New folder\\symptom_Description.csv')as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]:row[1]}
            dedcription_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open("C:\\Users\\SNEHA\\OneDrive\\Desktop\\New folder\\Symptom_severity.csv")as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getPreCautionDict():
    global precautionDictionary
    with open ('C:\\Users\\SNEHA\\OneDrive\\Desktop\\New folder\\symptom_precaution.csv')as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    name = input("Name: ")
    print("Your Name \n\t\t\t\t\t : ",name)
    print("Hello", name, "!")

def check_patterns(dis_list, inp):
    import re
    pred_list = []
    if not inp:
        return 0, []
    # treat input as literal (escape) and match case-insensitively
    try:
        regexp = re.compile(re.escape(inp), re.IGNORECASE)
    except re.error:
        regexp = re.compile(inp, re.IGNORECASE)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if pred_list:
        return 1, pred_list
    return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv(r'C:\Users\SNEHA\OneDrive\Desktop\New folder\Training.csv')
    X = df.iloc[:, :-1].copy()
    y = df['prognosis']
    # Encode string features in X
    for col in X.columns:
        if X[col].dtype == 'object':
            le_x = preprocessing.LabelEncoder()
            X[col] = le_x.fit_transform(X[col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return rf_clf.predict([input_vector])
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptoms] = index
    
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    
    return rf_clf.predict([input_vector]) 


def print_disease(node):
    print(node)
    node = node[0]
    print(len(node))
    val = node.nonzero()
    print(val)
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    conf_inp = int()
    while True:
        print("Enter the symptoms you are experiencing \n\t\t\t\t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_patterns(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num,")", it)
            if num != 0:
                print(f"Select the one you mean (0 - {num}): ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0
            
            disease_input = cnf_dis[conf_inp]
            break
            #print("Did you mean : ", cnf_dis, "?(yes/noo):", end+"")
            #conf_inp = input("")
            #if(conf_inp == "yes"):
            #break
        else:
            print("Enter valid symptoms.")

    while True:
        try:
            # clearer prompt so user doesn't think it's asking for a year/birth year
            num_days = int(input("How many days have you been experiencing these symptoms? (enter a number): "))
            if num_days < 0:
                print("Please enter a non-negative number of days.")
                continue
            break
        except ValueError:
            print("Invalid input — please enter the number of days as an integer (for example: 3).")
    
    def recurse(node, depth):
        indent = " "* depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]


            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_. value[node])
            #print("You may have" + present_disease)
            red_cols = reduced_data.columns

            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            #dis_list = list(symptoms_present)
            #if len(dis_list) != 0
            #print("Symptoms present "+ str(list(symptoms_present)))
            #print("Symptoms given" + str(list(syptoms_given)))
            print("Are you experiencing any")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms,"?:", end='')
                while True:
                    inp = input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("Provide proper answer i.e. (yes/no): ", end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)
            
            second_prediction = sec_predict(symptoms_exp)
            #print(seconf_prediction)
            calc_condition(symptoms_exp, num_days)
            if(present_disease[0] == second_prediction[0]):
                print("You may have", present_disease[0])

                print(dedcription_list[present_disease[0]])
                #readn(f"You may have {present_disease[0]})
                #readn(f"{description_list[present_disease[0]]})
            else:
                print("You may have ", present_disease[0], "or", second_prediction[0])
                print(dedcription_list[[present_disease[0]]])
                print(dedcription_list[second_prediction[0]])

            #print(description_list[present_disease[0]])
            precaution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list):
                print(i+1, ")", j)
            
            #confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            #print("confidence level is "+ str(confidence_level))
    recurse(0, 1)
getSeverityDict()
getDescription()
getPreCautionDict()
getInfo()
tree_to_code(clf, cols)
