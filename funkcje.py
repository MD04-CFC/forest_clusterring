def wykres2D(a,b):             
    import matplotlib.pyplot as plt   
      


def wykres3D(a,b, c):             
    import plotly.express as px


def histogram(a):  
    return 0        


def boxplot(a):   
    return 0    


def srednia(tablica):
    return sum(tablica)/len(tablica)

def opcja_1(tablica_jednego_typu, a, b):                                                          #_srodki_srednia
    return [ srednia(tablica_jednego_typu[a]), srednia(tablica_jednego_typu[b]) ]               




def wspolrzedne_punktu_z_danego_typu_2D(x, tablica_jednego_typu):
    return 1
    # return wyborpunktu[x]



def opcja_2(tablica_jednego_typu, a,b):                                                            #_srodki_z_danego_typu   --> wybor                   
    return [ wspolrzedne_punktu_z_danego_typu_2D(a, tablica_jednego_typu), wspolrzedne_punktu_z_danego_typu_2D(b, tablica_jednego_typu) ]         




import random
def opcja_3(tablica_jednego_typu,a,b):          #_srodki_losowe
    x = tablica_jednego_typu.sample(n=1).iloc[0]  

    return [x.iloc[0], x.iloc[1]]  
                                               
                     









