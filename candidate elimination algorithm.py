import pandas as pd

def more_general(h1, h2):
    return all(x == "?" or (x != "0" and (x == y or y == "0")) for x, y in zip(h1, h2))

def fulfills(example, hypothesis):
    return more_general(hypothesis, example)

def generalize_S(example, S):
    for i in range(len(S)):
        if not fulfills(example, S):
            S[i] = "?" if S[i] != example[i] else S[i]
    return S

def specialize_G(example, G, S):
    new_G = []
    for g in G:
        if fulfills(example, g): continue
        for i in range(len(g)):
            if g[i] == "?":
                for val in [example[i], "0"]:
                    new_g = g[:i] + [val] + g[i+1:]
                    if any(more_general(gh, new_g) for gh in G) and not fulfills(S, new_g):
                        new_G.append(new_g)
    return new_G

def candidate_elimination(training_data):
    attributes = training_data.columns[:-1]
    X, y = training_data.iloc[:, :-1].values, training_data.iloc[:, -1].values
    S, G = ["0"] * len(attributes), [["?"] * len(attributes)]
    
    for i, example in enumerate(X):
        if y[i] == "Yes":
            G = [g for g in G if fulfills(example, g)]
            S = generalize_S(example, S)
        else:
            S = ["?" if fulfills(example, s) else s for s in S]
            G = specialize_G(example, G, S)
    
    G = [g for g in G if any(more_general(g, s) for s in S)]
    return S, G

if __name__ == '__main__':
    file_path = "C:\\Users\\USER\\Documents\\ML\\breastcancer.csv"
    training_data = pd.read_csv(file_path)
    S, G = candidate_elimination(training_data)
    print("Most specific hypothesis S:", S)
    print("Most general hypotheses G:", G)
