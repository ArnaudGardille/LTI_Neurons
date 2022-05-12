# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:00:52 2021

@author: aforv
"""
import pickle
import os
import json
from re import S
from typing import List
import copy
import numpy as np
import time

import pandas as pd

from scipy import signal as sg
from scipy import integrate
from scipy.interpolate import interpn,interp1d,RegularGridInterpolator, Akima1DInterpolator
from scipy.optimize import minimize
import scipy.signal as sig
from scipy.optimize import least_squares
from numpy import linalg as la

from sklearn.metrics import mean_squared_error

# IMPORT THE MODEL API FROM WHICH YOUR MODEL MUST INHERITATE : 
try:
    from model_api import ModelApi
except:pass
try:
    from utilities.model_api import ModelApi
except:pass
try:
    from sources.utilities.model_api import ModelApi
except:pass
    
##################################################
## In this script, candidates are asked to implement two things:
#    1- the model, along with its training, prediction/inference methods;
#    2- the interface of this model with SKF evaluation and remote (AWS sagemaker) computation tools.
#
# See example notebook for an example of how to use this script
##################################################
## Author: MIA team
##################################################

def pred(x, t_e, entree):
    """
    Gives the response of a LTI system
    - at times t_e 
    - given an imput signal entree
    - given coefficients x
    """
    b = x[:int(len(x)/2)]
    a = x[int(len(x)/2):]
    H_1 = sig.lti(b,a)
    [t_s, sortie, xout] = H_1.output(entree,t_e)
    return np.array(sortie)

def err(x, t, entree, y):
    """ Computes the squared error between truth and prediction """
    return (pred(x, t, entree) - y)**2


def RMSE(real,prediction):
    """ Computes the mean squared error between truth and prediction """
    rmse = 0
    for k in range(len(real)):
        rmse += (real[k]-prediction[k])**2
    return rmse/len(real)

def CalcOffsetShrink(output_names,dic):
    ##Calcul des shrink (positif ou négatif) et des offset de chaque output

    offsets = {}
    shrinks = {}
    
    for name in output_names:
        key = list(dic.keys())[0]
        offset = dic[key][name][0]
        offsets[name]=offset
        
        try:
            shrink = (dic['input1'][name][-1]-offsets[name])/dic['input1']['Input'][-1]
        except:
            #print("erreur calcul shrink")
            shrink = 1

        shrinks[name] = shrink
    
    return offsets, shrinks

def removeOffsetShrink(dic, offsets, shrinks):
    ###Traitement des output
    dic_corrige = copy.deepcopy(dic)


    for key in dic_corrige.keys():
        for name in list(dic_corrige[key].keys())[2:]:
            dic_corrige[key][name] = dic[key][name] - offsets[name] 
            dic_corrige[key][name] = 1/shrinks[name]*dic_corrige[key][name]

    return dic_corrige

def addOffsetShrink(dic, offsets, shrinks):
    ###Traitement des output
    dic_corrige = copy.deepcopy(dic)
    for key in dic_corrige.keys():
        for name in list(dic_corrige[key].keys())[2:]:
            dic_corrige[key][name+str('_approx')] = shrinks[name]*dic_corrige[key][name+str('_approx')]
            dic_corrige[key][name+str('_approx')] = dic_corrige[key][name+str('_approx')]+offsets[name]
        
    return dic_corrige


def calcul_lipschitz(x):
    """
    Calcule la constante de Lipschitz d'une liste
    """
    max_L = 1
    L = 1
    Time_step = 0.001
    for k in range(len(x)-1):
        L = np.abs(x[k+1]-x[k])/np.abs(Time_step)
        if L > max_L :
            max_L = L    
    return max_L    



def calcul_changements_signe(x):
    """
    Calcule le nombre de changements de signe stricts d'une liste
    """
    compteur = 0
    
    for k in range(2,len(x)):
        if (x[k]*x[k-1]<0):
            compteur += 1
            
    return compteur 


def calcul_params_input(dic):
    """
    Calcule les paramètres de tous les inputs présents dans le dictionnaire
    """
    params_inputs = {}
    moyenne_lip = 0
    moyenne_chgmt_signe = 0
    for key in dic.keys() :
        x = dic[key]['Input']
        params_inputs[key] = [] ##[lip,chgmt_signe]
        lip = calcul_lipschitz(x)
        chgmt_signe = calcul_changements_signe(x)
        params_inputs[key].append(lip)
        params_inputs[key].append(chgmt_signe)
        moyenne_lip += lip/len(dic)
        moyenne_chgmt_signe += chgmt_signe/len(dic)

    ##normalisation des params
    """                   
    for key in dic.keys():
        params_inputs[key][0] = params_inputs[key][0]/moyenne_lip
        params_inputs[key][1] = params_inputs[key][1]/moyenne_chgmt_signe
    """   
    return params_inputs


def similitude_generale(poids,param1,param2):
    """
    Calcule la similitude entre deux liste paramètres | Grande similitude <=> paramètres proches
    """
    
    param_1 = np.array(param1)
    param_2 = np.array(param2)
    
    params_dif = param_1-param_2
    params_dif_norme = np.array([np.abs(param) for param in params_dif])
    norme = np.dot(poids,params_dif_norme)
    
    return 1/max([1e-15,norme])


def calcul_sorties_approx(poids,inputs_names,output_names,coeffs,params_inputs,d,offsets,shrinks):
    """
    Calcul les sorties de tous les inputs données en fonctions des autres outputs et de la similitude entre les inputs
    """
    Sorties = {}
    
    for key in d.keys():
        Sorties[key] = {}
        for name in output_names:
            Sorties[key][name+"_approx"] = np.zeros(len(d[key]['Time']))
                       

    for key in d.keys():
        inputs_calcul = []
        for inp in d.keys():
            if (inp != key) : inputs_calcul.append(inp)

        for name in output_names :
            similitudes_dic = {}
            for key_calcul in inputs_calcul:
                time1 = d[key]['Time']
                time2 = d[key_calcul]['Time']
                time = 0
                if len(time1) > len(time2) : time = time2
                else : time = time1
                similitudes_dic[key_calcul] = similitude_generale(poids,params_inputs[key],params_inputs[key_calcul])**2
            somme = sum(list(similitudes_dic.values()))

            for key_calcul in inputs_calcul:
                Time = d[key]['Time']
                Input = d[key]['Input']                       
                x = coeffs[name][key_calcul]
                sortie = pred(x,Time,Input)   ##Sortie prédite par le modèle entrainé sur l'input "key_calcul"
                Sorties[key][name+"_approx"] += sortie*(similitudes_dic[key_calcul]/somme)    ##Pondération par la similitude 
            Sorties[key][name+"_approx"] = Sorties[key][name+"_approx"]*shrinks[name]+offsets[name]
        Sorties[key] = pd.DataFrame(Sorties[key]) 
                       
    return Sorties
                       
def erreur_moyenne_simi(poids,inputs_names,output_names,coeffs,params_inputs,d,offsets,shrinks):                       
    """
    Calcul l'erreur moyenne de toutes sorties calculées par la fonction calcul_sorties_approx
    """
    Sorties = calcul_sorties_approx(poids,inputs_names,output_names,coeffs,params_inputs,d,offsets,shrinks)
                       
    mean_mse = 0
    mse_dic = {}

    for name in Sorties.keys():
        mse_dic[name] = 0
        for k in range(len(output_names)):
            maxi = np.max(np.abs(d[name][output_names[k]]))
            mse = mean_squared_error(d[name][output_names[k]]/maxi,Sorties[name][output_names[k]+"_approx"]/maxi)
            mean_mse += mse 
            mse_dic[name] += mse
            
            ##mean_rmse += rmse/(5*len(list(Sorties.keys())))
            
    return mean_mse/len(Sorties)
                       
    
    
    
    

"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL'S INTERFACE

Here you have to implement all the necessary methods as they are 
defined in the parent class ModelApi (cf file model_api.py).
These methods are used in higher levels scripts such as:
    - sagemaker_api.py that allows you (and us) to run training tasks on local or Amazon specific instances ;
    - calc_metrics.py/calc_metrics_on_sagemaker.py that allows you (and us) to compute the performance metrics of your solution, given your model definition (this file) and a test dataset as input;
"""
class MyModel(ModelApi):

    def __init__(self, degre=2, nbOutputs=5, epochs_1 = 80, epochs_2 = 15, Ndecim = 1, **model_kwargs):
        self.model_kwargs = model_kwargs
        self.epochs_1 = epochs_1
        self.epochs_2 = epochs_2
        self.degre = degre
        self.coeffs = {}
        self.nbOutputs = nbOutputs
        ###Paramètres fixes associés aux 5 systèmes
        self.shrinks = {'Output1':-1.0662465662259057,'Output2':0.49999724092332437,'Output3':-0.499948306241051,'Output4':0.8293801429644225,'Output5':-1.1137867977024063} 
        self.offsets = {'Output1':-2.4641212987881885e-07,'Output2':0.4999999999746566,'Output3':0.4999999999772357,'Output4':1.011922833586301,'Output5':1.009829699836818}
        self.poids = np.array([-0.00616706,1.1635385])
        self.params_inputs_train = {}
        self.approx = {}
        self.output_names = []
        self.dic_train = {}
        self.Ndecim = Ndecim
        
                       
    def fit(self, xs: List[np.ndarray], ys: List[np.ndarray], timeout=36000, verbose=False):  ##prend en entrée un ou plusieurs inputs et leur réponse
        
        print("--- Data processing ---")
        
        
        
        self.nbOutputs = len(ys)
     
        output_names = ["Output"+str(k) for k in range(1,self.nbOutputs+1)]
        self.output_names = output_names
        
        ##Preprocessing data
        ##Temps de séparation des signaux
        ##Nouvel input si temps nul >= 700
        
        seuil = int(700/self.Ndecim)
        indice = 0
        compteur = 0
        compteur_total = 0
        liste_temps = []
        liste_couples_temps = []
        
        while compteur_total < len(xs[0])-2:
            while (compteur < seuil) and (compteur_total < len(xs[0])-2):
                compteur_total += 1
                if xs[0][compteur_total] == 0:
                    compteur += 1
                else:
                    compteur = 0
            diff_temps = 0
            while (xs[0][compteur_total+diff_temps]==0) and (compteur_total+diff_temps < len(xs[0])-2):
                diff_temps += 1
            liste_temps.append(int((compteur_total-compteur)+(compteur+diff_temps)/2))
            liste_couples_temps.append([compteur_total-compteur,compteur_total+compteur+diff_temps])
            compteur = 0
            compteur_total = compteur_total+diff_temps
        
        if len(liste_temps) > 0 :
            if liste_temps[0] <= 3*seuil : 
                liste_temps.remove(liste_temps[0])
                liste_couples_temps.remove(liste_couples_temps[0])
            if (len(liste_temps)>0) and (len(xs[0])-liste_temps[-1] >= 3*seuil) : 
                liste_temps.remove(liste_temps[-1])
                liste_couples_temps.remove(liste_couples_temps[-1])

            
        
        nb_signaux = max([0,len(liste_temps)])
        
        input_names = ["input"+str(k) for k in range(nb_signaux)]
        
        ##On met les données sous la forme d'un dictionnaire, plus simple pour la suite
        dic_train = {}
        prev_time = 0
        for i in range(len(input_names)) :
            ##n = liste_temps[i]
            n = int(liste_couples_temps[i][0]+0.1*(liste_couples_temps[i][1]-liste_couples_temps[i][0]))  ## On supprimes les longs moments où les signaux sont nuls
            time = np.arange(n-prev_time)*0.001
            dic_train[input_names[i]] = {}
            dic_train[input_names[i]]['Time'] = time
            dic_train[input_names[i]]['Input'] = xs[0][prev_time:n]
            for j in range(len(output_names)) : 
                dic_train[input_names[i]][output_names[j]] = ys[j][prev_time:n]
            prev_time = int(liste_couples_temps[i][1]-0.1*(liste_couples_temps[i][1]-liste_couples_temps[i][0]))            
            
        self.dic_train = copy.deepcopy(dic_train)

        Compteur = 0               
        
        for output in output_names :
            self.coeffs[output]={}
        
        for key in input_names:
            self.approx[key] = {}    
        
        print("--- Fit started ---")
                
        
        ###Normalisation des outputs pour utiliser des systèmes LTI

        dic_train_corr = removeOffsetShrink(dic_train, self.offsets, self.shrinks)            
                        
        N_total = len(output_names)*len(input_names)
        Compteur = 0        
        
        ##Calcul des coeffs fction de transferts 
        
        print("Calcul des " +str(N_total) + " liste(s) de coefficients des systèmes linéaires, itérations maximales de chacune des minimisations : " +str(self.epochs_1))
        
        for Input in input_names :
            for Output in output_names :
                x_train = dic_train[Input]['Input']
                y_train = dic_train_corr[Input][Output]

                n = x_train.size
                time_train = np.arange(n)*0.001                 
                
                x0 = np.ones(2*self.degre)
                x = np.arange(1,2*self.degre+1)
                res1 = least_squares(err, x0, bounds=(-3, 3), args=(time_train,x_train,y_train), verbose=verbose, ftol=1e-15, gtol=1e-15, xtol = 1e-15, max_nfev = self.epochs_1)
                
                x = res1.x
               
                self.approx[Input][Output+"_approx"] = pred(x,time_train,x_train)   ###Réponse du système LTI approximant
                self.approx[Input][Output+"_approx"] = self.approx[Input][Output+"_approx"]*self.shrinks[Output]+self.offsets[Output] 
                
                self.coeffs[Output][Input] = x
                
                ###"Dénormalisation" de la réponse calculée
                Compteur += 1
                if verbose:
                    print("Avancement calcul des fonctions de transfert : " + str(round(Compteur/N_total*100)) +"%")
                    
                       
        ##Calcul params_inputs
        params_inputs = calcul_params_input(dic_train)

        self.params_inputs_train = params_inputs
        
        
        ##optimisation du poids, seulement si au moins 2 inputs
        
        if len(dic_train) > 1 :
        
            print("--------------")             
            print("Optimisation des poids dans la norme de similitude, itérations de l'optimisation : " +str(self.epochs_2))               


            x0 = self.poids
            x = np.arange(1,3)
            res = least_squares(erreur_moyenne_simi, x0, args=(input_names,output_names,self.coeffs,params_inputs,dic_train,self.offsets,self.shrinks), verbose=2, ftol=1e-15, gtol=1e-15, xtol = 1e-15, max_nfev = self.epochs_2)
            self.poids = res.x
            print("Poids similitude : " +str(self.poids))
            print("Normalized MSE sum moyenne sur les signaux prédits en fonction des autres : " + str(erreur_moyenne_simi(self.poids,input_names,output_names,self.coeffs,params_inputs,dic_train,self.offsets,self.shrinks)))
                     
                     
                                       

    @classmethod
    def get_sagemaker_estimator_class(self):
        """
        return the class with which to initiate an instance on sagemaker:
        e.g. SKLearn, PyTorch, TensorFlow, etc.
        by default - use SKLearn image.
        """
        
        from sagemaker.pytorch import PyTorch
        framework_version = '1.8.0'
        
        return PyTorch,framework_version

    def predict_timeseries(self, x) : ##-> {output_name : np.array()}

        Sortie = {'input':{}}
        for name in self.output_names :
            Sortie['input'][name+"_approx"] = np.zeros(len(x))
        n = len(x)
        time1 = np.arange(n)*0.001
                     
        for name in self.output_names :
            similitudes_dic = {}
            for key_calcul in self.params_inputs_train.keys():
                time2 = self.dic_train[key_calcul]['Time']
                time = 0
                if len(time1) > len(time2) : time = time2
                else : time = time1

                params = [calcul_lipschitz(x),calcul_changements_signe(x)]
                similitudes_dic[key_calcul] = similitude_generale(self.poids,params,self.params_inputs_train[key_calcul])**2
            somme = sum(list(similitudes_dic.values()))

            for key_calcul in self.params_inputs_train.keys():                      
                coeffs = self.coeffs[name][key_calcul]
                sortie = pred(coeffs,time1,x) 
                Sortie['input'][name+"_approx"] += sortie*(similitudes_dic[key_calcul]/somme)
            Sortie['input'][name+"_approx"] = Sortie['input'][name+"_approx"]*self.shrinks[name]+self.offsets[name]
        
        Sortie_array = np.transpose(list(Sortie['input'].values()))
                    
        return Sortie_array
        
  


    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        filenames = ["degre", "coeffs", "nbOutputs", "shrinks", "offsets", "poids", "params_inputs_train", "approx", "output_names"]
    
        path = os.path.join(model_dir, "degre")
        np.save(path, self.degre)

        path = os.path.join(model_dir, "coeffs") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.coeffs, a_file)
        a_file.close()  

        path = os.path.join(model_dir, "nbOutputs")
        np.save(path, self.nbOutputs)

        path = os.path.join(model_dir, "shrinks") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.shrinks, a_file)
        a_file.close()  

        path = os.path.join(model_dir, "offsets") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.offsets, a_file)
        a_file.close()  

        path = os.path.join(model_dir, "poids")
        np.save(path, self.poids)        

        path = os.path.join(model_dir, "Ndecim")
        np.save(path, self.Ndecim) 
        
        path = os.path.join(model_dir, "params_inputs_train") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.params_inputs_train, a_file)
        a_file.close()       

        path = os.path.join(model_dir, "approx") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.approx, a_file)
        a_file.close()           

        path = os.path.join(model_dir, "output_names")
        np.save(path, self.output_names)         
                
        path = os.path.join(model_dir, "dic_train") ##dic
        a_file = open(path, "wb")
        pickle.dump(self.dic_train, a_file)
        a_file.close()            
        
    @classmethod
    def load(cls, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        model = cls() 

        path = os.path.join(model_dir, "degre.npy")
        model.degre = np.load(path, allow_pickle=True)
        
        path = os.path.join(model_dir, "Ndecim.npy")
        model.Ndecim = np.load(path, allow_pickle=True)

        path = os.path.join(model_dir, "coeffs")
        a_file = open(path, "rb")
        model.coeffs = pickle.load(a_file)
        a_file.close()

        path = os.path.join(model_dir, "nbOutputs.npy")
        model.nbOutputs = np.load(path, allow_pickle=True)

        path = os.path.join(model_dir, "shrinks")
        a_file = open(path, "rb")
        model.shrinks = pickle.load(a_file)
        a_file.close()

        path = os.path.join(model_dir, "offsets")
        a_file = open(path, "rb")
        model.offsets = pickle.load(a_file)
        a_file.close()

        path = os.path.join(model_dir, "poids.npy")
        model.poids = np.load(path, allow_pickle=True)        

        path = os.path.join(model_dir, "params_inputs_train")
        a_file = open(path, "rb")
        model.params_inputs_train = pickle.load(a_file)    
        a_file.close()  

        path = os.path.join(model_dir, "approx")
        a_file = open(path, "rb")
        model.approx = pickle.load(a_file)    
        a_file.close()

        path = os.path.join(model_dir, "output_names.npy")
        model.output_names = np.load(path, allow_pickle=True)         

        path = os.path.join(model_dir, "dic_train")
        a_file = open(path, "rb")
        model.dic_train = pickle.load(a_file)  
        a_file.close()         
        
        return model

    @classmethod
    def create_model(cls, gpu_available: bool = False, **kwargs):
        return cls(**kwargs)

    @property
    def description(self):
        team_name = 'MIA'
        email = 'arnaud.gardille@gmail.com'
        model_name = 'LTI_v2'
        affiliation = 'Université Paris-Saclay'
        description = 'This is a simple LTI model that supports 1 or more inputs and 1 to 5 corresponding outputs'
        technology_stack = 'Scipy'
        other_remarks = ''

        return dict(team_name=team_name,
                    email=email,
                    model_name=model_name,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)