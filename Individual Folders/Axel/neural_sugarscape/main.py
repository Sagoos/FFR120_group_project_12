import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sugarscape




#board parameters
n_side=200
spotpos = np.random.randint(0, n_side, (4, 2))
spotwidths = np.ones(len(spotpos)) * 40
maxsugar=10
sugargrowth=0.004
#ant parameters-----------------------------------------
vis=3
met=3
nants=70
#initializing
brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth)


#loading in the starting agent parameters
dub=np.load('files/w_run1_iteration_1999_.npy')
tet=np.load('files/theta_run1_iteration_1999_.npy')
ants=sugarscape.ants(vis,met,nants,w=dub,theta=tet)
#ants=sugarscape.ants(vis,met,nants)
antpos=np.random.randint(0,n_side,(nants,2))


print('total sugar consumption/total sugar growth: ',met*nants/(sugargrowth*np.sum(brd.currentsugar)))
#GA parameters
epochs=2000
runtime=500
pmut=0.07
sigma_mut=np.std(ants.w)/3

#the iterations for which the agent paramters should be saved
iterations_to_save=np.array([*np.arange(0,epochs,10),epochs-1])
l=n_side-1

#the iteration the training is starting from
startnum=1999

for e in range(epochs):
    spotpos = np.random.randint(0, n_side, (4, 2))
    spotwidths = np.ones(len(spotpos)) * 40
    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth)
    print(f'\rtraining {e/epochs*100}% done',end='')
    if e in iterations_to_save:
        np.save(f'files/w_run1_iteration_{e+startnum}_',ants.w)
        np.save(f'files/theta_run1_iteration_{e+startnum}_', ants.theta)

    for i in range(runtime):
        #get moves of all the living ants
        antpos[ants.alive]=ants.get_newpos(board=brd.currentsugar,antpos=antpos,l=l)
        #let living ants move and eat
        ants.eat(brd.currentsugar,antpos)
        #hunger
        ants.hunger(i)
        #grow sugar back
        brd.growsugar()
        #break if alla ants dead
        if np.all(~ants.alive):
            break

    #updating population
    fitness=sugarscape.fitnessess(ants.lifetimes,runtime,ants.alive,ants.sugarlevels)
    indices=sugarscape.roullette_wheel(fitness)
    new_w=np.zeros_like(ants.w)
    new_theta=np.zeros_like(ants.theta)
    #performing crossover
    for i in range(len(indices)//2):
        new_w[[i*2,i*2+1]]=sugarscape.crossover(ants.w[indices[2*i]],ants.w[indices[2*i+1]])
        new_theta[[i*2,i*2+1]]=sugarscape.crossover(ants.theta[indices[2*i]],ants.theta[indices[2*i+1]])
    #performing muation
    sugarscape.mutate(new_w,pmut,sigma_mut)
    sugarscape.mutate(new_theta,pmut,sigma_mut)
    #updating parameters
    ants.w=new_w
    ants.theta=new_theta
    #resetting
    brd.reset()
    ants.reset()
    antpos=np.random.randint(0,n_side,(nants,2))

