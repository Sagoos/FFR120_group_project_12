import matplotlib.pyplot as plt
import numpy as np
import sugarscape
import os
from scipy.signal import savgol_filter


#board parameters
n_side=200
spotpos = np.random.randint(0, n_side, (4, 2))
spotwidths = np.ones(len(spotpos)) * 40

dangerpos=np.random.randint(0,n_side,(3,2))
dangerwidth=np.ones(len(dangerpos))*20
maxdanger=15

maxsugar=10
sugargrowth=0.004
#ant parameters-----------------------------------------
vis=2
met=3
nants=70

#GA parameters
runtime=1000
l=n_side-1



def runmodel(w,theta):
    x=np.random.randint(0,n_side,(nants,2))

    spotpos=np.random.randint(0,n_side,(4,2))
    dangerpos=np.random.randint(0,n_side,(3,2))

    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth,dangerpos,dangerwidth,maxdanger)
    ants = sugarscape.ants(vis, met, nants,w=w,theta=theta)
    fractions=np.ones((runtime,nants))*np.nan
    for i in range(runtime):

        # get moves of all the living ants
        bestsug =ants.get_deterministic_sugar(board=brd.currentsugar,antpos=x,l=l)
        x[ants.alive] = ants.get_newpos(board=brd.currentsugar,dboard=brd.dangerlevels, antpos=x, l=l)
        fractions[i,ants.alive]=brd.currentsugar[tuple(x[ants.alive].T)]/bestsug

        # let living ants move and eat
        ants.eat(brd.currentsugar, x)
        # hunger
        ants.hunger(i,brd.dangerlevels,x)
        # grow sugar back
        brd.growsugar()
        if np.all(~ants.alive):
            break

    lifetimes=ants.lifetimes
    fitnesses=sugarscape.fitnessess(ants.lifetimes,runtime,ants.alive,ants.sugarlevels)
    fractions=np.nanmean(fractions,axis=0)
    fractions[fractions==0]=np.nan
    livingfraction=np.sum(ants.alive)/nants
    assert np.all(np.isnan(lifetimes)==ants.alive)
    return fitnesses,fractions,lifetimes,livingfraction


def run_eval():
    runtime=1000
    directory = "files3"
    files=os.listdir(directory)
    s=[f.split('_') for f in files]
    s = [x for x in s if x[0] != "run1"]
    runs=[int(f[1].replace("run","").strip()) for f in s]
    iters=[int(f[3]) for f in s]

    iters=np.array(iters)
    iters.sort()
    #code for deleting all network parameters after a given iteration
    '''
    iters2delete=iters[iters>21900]
    for i in range(len(iters2delete)//2):
        os.remove(f'files3/w_run1_iteration_{iters2delete[2*i]}_.npy')
        os.remove(f'files3/theta_run1_iteration_{iters2delete[2*i+1]}_.npy')
    #stops the program
    assert 1==0
    '''
    #initilizing
    fitnesses=np.zeros((len(iters),nants))
    fractions=np.zeros((len(iters),nants))
    lifetimes=np.zeros((len(iters),nants))
    livingfractions=np.zeros(len(iters))

    for i in range(len(iters)):
        print(f'\r{i/len(iters)*100}% done',end='')
        w=np.load(f'files3/w_run1_iteration_{iters[i]}_.npy')
        theta=np.load(f'files3/theta_run1_iteration_{iters[i]}_.npy')
        fitnesses[i],fractions[i],lifetimes[i],livingfractions[i]=runmodel(w,theta)

    np.save('files3/run1_fitnesses',fitnesses)
    np.save('files3/run1_fractions',fractions)
    np.save('files3/run1_lifetimes',lifetimes)
    np.save('files3/run1_livingfractions.npy',livingfractions)

#run_eval()


fitnesses=np.load('files3/run1_fitnesses.npy')
fractions=np.load('files3/run1_fractions.npy')
lifetimes=np.load('files3/run1_lifetimes.npy')
livfrac=np.load('files3/run1_livingfractions.npy')


lifetimes[np.isnan(lifetimes)]=runtime



files=os.listdir('files3')
s=[f.split('_') for f in files]
s=[x for x in s if x[0]!="run1"]
iters=[int(f[3]) for f in s]


iters=np.array(iters)
indices=np.argsort(iters)
iters=iters[indices]

print(iters[-1])
print(fitnesses.shape)
print(fractions.shape)
print(livfrac.shape)

meanfrac=np.mean(fractions,axis=1)
meanfit=np.mean(fitnesses,axis=1)
meanlife=np.mean(lifetimes,axis=1)

smeanfrac=savgol_filter(meanfrac, window_length=50, polyorder=2)
smeanfit=savgol_filter(meanfit, window_length=50, polyorder=2)
smeanlife=savgol_filter(meanlife, window_length=50, polyorder=2)
smeanlivfrac=savgol_filter(livfrac, window_length=50, polyorder=2)


fig,axs=plt.subplots(1,3)

for i in range(70):
    axs[0].plot(iters,fractions[:,i],color='lightblue',alpha=0.05)
    axs[1].plot(iters,fitnesses[:,i],color='orange',alpha=0.05)
    axs[2].plot(iters,lifetimes[:,i],color='lightgreen',alpha=0.05)

axs[0].plot(iters,smeanfrac,color='k',label='smoothed population mean fraction')
axs[1].plot(iters,smeanfit,color='k',label='smoothed population mean fitness')
axs[2].plot(iters,smeanlife, color='k', label='smoothed population mean lifetime')

'''
axs[0].plot(iters,np.polyval(fracfit,iters),color='k',label='mean fraction')
axs[1].plot(iters,np.polyval(fitfit,iters),color='k',label='mean fitness')
axs[2].plot(iters, np.polyval(lifefit,iters), color='k', label='mean lifetime')
'''

axs[0].set_title('fractions')
axs[1].set_title('fitnesses')
axs[2].set_title('lifetimes')

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].grid()
axs[1].grid()
axs[2].grid()

plt.show()

plt.plot(iters,smeanlivfrac)
plt.show()

