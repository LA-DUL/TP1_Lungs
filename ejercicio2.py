import numpy as np
import os
import matplotlib.pyplot as plt
# Módulo para manejo de imágenes
from PIL import Image
#img = Image.open(img_path).convert("L") # abrir una imagen en grises
#img_matrix = np.array(img) # convertir imagen a numpy
#img_rotada = np.rot90(img_matrix, 2) # imagen rotada 180° (2 * 90°)

from pca_funciones_auxiliares import img2blocks, blocks2img
#dado un valor de p, cómo hago una simulación?
def get_labels_test():
    labels = {
        #son un monton de imagenes. cómo son solo 2 estados puedo armar un dict con key 1 y 0 y guardar cómo valores a cdu
        0: [],
        1: [],
    }
    with open(r'C:\Users\leond\OneDrive\Escritorio\UdeSA\2026_segundo_semestre\inf_est\TP1\dataset_tp1\dataset_tp1\test_labels.csv', 'r') as arch:
    #tray = arch.read()
    
        next(arch)
        for fila in range(468):
            linea = next(arch)
            d = linea.strip().split(',')
            labels[int(d[1])].append(d[0])
        return labels
def get_img_ord_arch():
    with open(r'C:\Users\leond\OneDrive\Escritorio\UdeSA\2026_segundo_semestre\inf_est\TP1\dataset_tp1\dataset_tp1\test_labels.csv', 'r') as arch:
    #tray = arch.read()
        img_in_order = []
        next(arch)
        for fila in range(468):
            linea = next(arch)
            d = linea.strip().split(',')
            img_in_order.append(d[0])
        return img_in_order
#Para cda imagen la proba de que este dada vuelta es tipo una Bernouli con proba p
#si 1, que esta rotada, la tenemos que dar vuelta 180°
#si p es chico deberían haber pocas imagenes, si grande, muchas
#para el 2 a necesito las componentes. Ya lo calculamos (supuestamente) en el 1b, mepa que deberia dar parecido.
# si no tiene nada que ver, prob que tmb ande mal el estimador. Seria un problema, pero sumeria al tp XD
# pero para el 2 lo debería calcular con que imagenes fueron pertubadas y con su coord en PCA
#print(get_labels_test())


 

def perturbar_test(X, p):
    base = r'C:\Users\leond\OneDrive\Escritorio\UdeSA\2026_segundo_semestre\inf_est\TP1\dataset_tp1\dataset_tp1\test'
    perturbar_test = []
    rotadas = []
    for Xi in X :
        #Xi serian las imagenes del testog
        #Xi += '.png' #al parecer ya incluye png
        img_path = os.path.join(base, Xi)
        img = Image.open(img_path).convert("L") # abrir una imagen en grises
        img_matrix = np.array(img) # convertir imagen a numpy
        choice = np.random.choice([True, False], p=[p, 1-p])
        if choice:
            rotadas.append(Xi)
            img_rotada = np.rot90(img_matrix, 2)
        else:
            img_rotada = img_matrix
        perturbar_test.append(img_rotada)
    return perturbar_test, rotadas
    


def PCA_LR(smt, K=2):
    #la implementación del 1
    #funcion magica que me devuelve el accuracy
    '''
    Necesito que ka PCA_LR me de el PC1 y PC2 de cada imagen (las coords. LO que tenes antes d epredecir con LR)
    tmb para el 2 OJO que es K=2
    
    '''
    print('Todo bien!')
    return 1

def sim(img_list, probabilidad):
    turbio, rot = perturbar_test(img_list, probabilidad)
    #si no tiene nada que ver el 2a con el 1b revisamos con rot
    accuracy = PCA_LR(turbio)
    return accuracy

def get_accuracy(n, proba, img):
    #hago ej2 a n veces
    accuracies = [] #A_1 - A_Nmc
    for _ in range(n):
        accuracies.append(sim(img, proba)) #Ai
    return accuracies


def ej2(Nmc, A0, m=3):
    #el A0 lo tienen que calcular con PCA+LR con K=2 !
    #label_test = get_labels_test()
    #img_list = label_test[0] + label_test[1]
    #si ustedes entrenan el PCA+LR con una lista dif, pasenmela asi hacemos lo mismo
    #n = np.arange(Nmc)
    img_list = get_img_ord_arch()
    E_Aps = []
    ps = [0.1, 0.5, 0.8] #para test
    #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #para final
    prob_delta = []
    for i in range(m): 
        p = ps[i]
        acc = get_accuracy(Nmc, p, img_list)
        #Hacemos monte carlo 1 vez para conseguir accuracies y reusamos para 2b-d
        #2b, promedio (media muestral)
        esperanza_Ap = sum(acc)/Nmc
        E_Aps.append(esperanza_Ap) 
        #2c prob que supera tol
        I = []
        for Ai in acc:
            if(A0-Ai) > 0.1:
                #delta_tol = 0.1
                I.append(1)
                #win!
            else:
                I.append(0)
                #loss :c
        pd=sum(I)/Nmc
        prob_delta.append(pd)
        #2d histograma
        plt.hist(acc, bins=30, color='blue', edgecolor='black', density=True, label='2D')
        plt.xlabel('Ap', fontsize=12)
        plt.ylabel('p', fontsize=12)
        plt.title(f'2d para montecarlo {m}°')
    #-----Grafo B-----
    plt.figure(figsize=(8,5))
    plt.plot(ps, E_Aps, "r^-", label='2B') #dps grafico
    plt.xlabel('probabilidades', fontsize=12)
    plt.ylabel('Esperanzas m', fontsize=12)
    plt.title('2B Valor medio del accuracy', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-----Grafo C-----
    plt.figure(figsize=(8,5))
    plt.plot(ps, prob_delta, "bo-", label='2C')
    plt.xlabel('probabilidades', fontsize=12)
    plt.ylabel('prob_delta', fontsize=12)
    plt.title('2C probabilidad perdida supere tol', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

