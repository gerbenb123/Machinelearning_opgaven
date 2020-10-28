import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plotNumber(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    new_vector = np.reshape(nrVector, (20, 20), order='A').T
    plt.matshow(new_vector)
    plt.show()

# ==== OPGAVE 2a ====

def one_sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    if isinstance(z, int):
        return one_sigmoid(z)
    if z.shape[0] > z.shape[1]:
        z = z.T

    return np.array([one_sigmoid(i) for i in z])[0]



# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    width = np.max(y)
    new_y = y.copy()
    new_y[new_y == width] = 0
    rows = [i for i in range(m)]
    data = [1 for _ in range(m)]
    y_vec = csr_matrix((data, (rows, new_y.T[0])), shape=(m, width)).toarray()

    return y_vec



# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predictNumber(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.
    ones = np.ones((1, X.shape[0])).T
    a1 = np.c_[ones, X].T
    z2 = np.dot(Theta1, a1)
    a2 = np.array([sigmoid(np.array(i, ndmin=2)) for i in z2])
    ones = np.ones((1, z2.shape[1]))
    a2 = np.r_[ones, a2]
    z3 = np.dot(Theta2, a2)

    return np.array([sigmoid(np.array(i, ndmin=2)) for i in z3]).T


# ===== deel 2: =====
def cost_of_one(prediction, y):
    return np.multiply(-y, np.log(prediction)) - np.multiply((1 - y), np.log(1 - prediction))

def computeCost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix.
    predictions = predictNumber(Theta1, Theta2, X)
    actual = get_y_matrix(y, X.shape[0])
    differences = predictions - actual

    cost = np.array([np.sum(cost_of_one(predictions[i], actual[i])) for i in range(len(differences))], ndmin=2).T
    return sum(cost) / len(cost)


# ==== OPGAVE 3a ====
def sigmoidGradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

# ==== OPGAVE 3b ====
def nnCheckGradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    # Deze heb ik hernoemd naar hoe ze in de reader staan.
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    m = X.shape[0] # voorbeeldwaarde; dit moet je natuurlijk aanpassen naar de echte waarde van m
    new_y = get_y_matrix(y, X.shape[0])

    for i in range(m):
        #xi zijn de activaties van de pixels van X[i] (1 picture)
        xi = np.array([X[i]], ndmin=2)

        a1 = np.c_[1, xi]
        z2 = np.dot(Theta1, a1.T)
        a2 = np.c_[1, np.array([sigmoid(np.array(i, ndmin=2)) for i in z2]).T]

        a3 = predictNumber(Theta1, Theta2, xi)
        delta_3 = a3 - new_y[i]

        differential = np.c_[1, np.array([sigmoidGradient(z2)], ndmin=2)]
        delta_2 = np.dot(Theta2.T, delta_3.T).T * differential
        Delta2 += np.dot(delta_3.T, a2)
        Delta1 += np.dot(np.array(np.delete(delta_2.T, 0), ndmin=2).T, a1)

    Delta1_grad = Delta1 / m
    Delta2_grad = Delta2 / m
    
    return Delta1_grad, Delta2_grad
