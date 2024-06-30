import numpy as np

def find_s_algorithm(training_data):
    
    hypothesis = training_data[0][:-1]  

    for instance in training_data:
        if instance[-1] == 'Yes':  
            for i in range(len(hypothesis)):
                if instance[i] != hypothesis[i]:
                    hypothesis[i] = '?' 
    return hypothesis

if __name__ == '__main__':
   
    training_data = np.array([
        ['big','red','circle','no'],
        ['small','red','triangle','no'],
        ['small','red','circle','yes'],
        ['big','blue','circle','no'],
        ['small','blue','circle','yes']
    ])

   
    hypothesis = find_s_algorithm(training_data)

   
    print("The maximally specific hypothesis is:", hypothesis)
