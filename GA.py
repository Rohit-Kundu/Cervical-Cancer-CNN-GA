import numpy as np
import pandas as pd
import time
import argparse

tic=time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--csv_name', type=str, required = True, help='Name of csv file- Example: SpectEW.csv')
parser.add_argument('--csv_header', type=str, default = 'no', help='Does csv file have header?: yes/no')
parser.add_argument('--pop_size', type=int, default = 100, help='Population size for Genetic Algorithm.')
parser.add_argument('--generations', type=int, default = 50, help='Number of Generations to run the Genetic Algorithm')
parser.add_argument('--mutation', type=int, default = 6, help='Percentage of mutation in the Genetic Algorithm')
args = parser.parse_args()

root = "../"
if root[-1]!='/':
    root+='/'
csv_path = args.csv_name
if csv_path.split('.')[-1]!='csv':
    csv_path = csv_path+".csv"
if args.csv_header.lower()=="yes":
    df = np.asarray(pd.read_csv(root+csv_path))
else:
    df = np.asarray(pd.read_csv(root+csv_path,header=None))
data = df[:,0:-1]
target = df[:,-1]

#Genetic algorithm
import matplotlib.pyplot as plt
import random
import csv
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from utils import *

data_inputs = data
data_outputs = target

#No. of classes
unique=np.unique(data_outputs)
num_classes=unique.shape[0]
classes=[]
for i in range(num_classes):
    classes.append('Class'+str(i+1))

num_samples = data_inputs.shape[0]
num_feature_elements = data_inputs.shape[1]
print("Total Number of Samples: ",num_samples)
print("Number of features: ",num_feature_elements)

#For KFold Cross Validation
from numpy import array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
Fold = 5

kfold = KFold(Fold, True, 1)

#The main program
sol_per_pop = args.pop_size # Population size
num_parents_mating = (int)(sol_per_pop*0.5) # Number of parents inside the mating pool.
num_mutations = (int)(sol_per_pop*num_feature_elements*args.mutation/100) # Number of elements to mutate.
num_generations = args.generations #Number of generations in each fold

print("Population size: {}".format(sol_per_pop))
print("Number of parents inside mating pool: {}".format(num_parents_mating))
print("Number of elements to mutate: {}".format(num_mutations))

test=0
f=0
fold_acc_list = []
best_outputs_test = []
best_num_feat = data.shape[1]

for train_index, test_index in kfold.split(data_inputs):
    fold_acc = 0
    
    train_data, train_label = np.asarray(data_inputs[train_index]), np.asarray(data_outputs[train_index])
    test_data, test_label = np.asarray(data_inputs[test_index]), np.asarray(data_outputs[test_index])

    # Defining the population shape.
    pop_shape = (sol_per_pop, num_feature_elements)

    # Creating the initial population.
    new_population = np.random.randint(low=0, high=2, size=pop_shape)

    for generation in range(num_generations):
        print("\nFold: {}, Generation : {}".format(f+1,generation+1))
        # Measuring the fitness of each chromosome in the population.
        fitness,  prediction_test, decision_test = cal_pop_fitness(new_population, train_data, train_label,test_data, test_label)
        
        best_outputs_test.append(np.max(fitness))
        
        print("Best test result : ", best_outputs_test[-1])
        best_match_idx = np.where(fitness == np.max(fitness))[0]
        best_match_idx = best_match_idx[0]
        best_solution = new_population[best_match_idx, :]
        num_feat_selected = np.where(best_solution == 1)[0].shape[0]
        print("Number of selected features: ", num_feat_selected)

        #Save the best combination for printing later
        if best_outputs_test[-1]>test:
            best_match_idx = np.where(fitness == np.max(fitness))[0]
            best_match_idx = best_match_idx[0]
            
            best_solution = new_population[best_match_idx, :]
            best_solution_indices = np.where(best_solution == 1)[0]
            indi=best_solution_indices.tolist()
            best_sol=[]
            for i in range(num_feature_elements):
                if i in indi:
                    best_sol.append(1)
                else:
                    best_sol.append(0)
            best_sol=np.array(best_sol)
            best_solution_num_elements = best_solution_indices.shape[0]
            best_solution_fitness = fitness[best_match_idx]
            predict_test=prediction_test

            t_label=test_label

            cl_name=[]
            for i in range(num_classes):
                cl_name.append(i+1)
            #plot_roc(test_label,decision_test, cl_name, f+1, caption='ROC')

            test=best_outputs_test[-1]
            best_num_feat = num_feat_selected
            gen=generation+1
            fold=f+1
            
        elif best_outputs_test[-1]==test and num_feat_selected<best_num_feat:
            best_match_idx = np.where(fitness == np.max(fitness))[0]
            best_match_idx = best_match_idx[0]
            
            best_solution = new_population[best_match_idx, :]
            best_solution_indices = np.where(best_solution == 1)[0]
            indi=best_solution_indices.tolist()
            best_sol=[]
            for i in range(num_feature_elements):
                if i in indi:
                    best_sol.append(1)
                else:
                    best_sol.append(0)
            best_sol=np.array(best_sol)
            best_solution_num_elements = best_solution_indices.shape[0]
            best_solution_fitness = fitness[best_match_idx]
            predict_test=prediction_test

            t_label=test_label

            cl_name=[]
            for i in range(num_classes):
                cl_name.append(i+1)
            #plot_roc(test_label,decision_test, cl_name, f+1, caption='ROC')

            test=best_outputs_test[-1]
            best_num_feat = num_feat_selected
            gen=generation+1
            fold=f+1

        if best_outputs_test[-1]>fold_acc:
            fold_acc = best_outputs_test[-1]

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations=num_mutations)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    fold_acc_list.append(fold_acc)
    f+=1
        
# Getting the best solution after iterating finishing all generations.
print("\n\n\nBest scores obtained at Fold {}, Generation: {}".format(fold,gen))
print("Number of selected elements: ", best_solution_num_elements)
print("Best Accuracy: ", best_solution_fitness)

toc=time.time()
print("\nComputation time is {} minutes".format((toc-tic)/60))

