import matplotlib.pyplot as plt
import numpy as np
import sugarscape
import os

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

#GA parameters
runtime=1000
l=n_side-1



def runmodel(w,theta):
    x=np.random.randint(0,n_side,(nants,2))
    x_ret=np.zeros((runtime,nants,2))
    im_ret=np.zeros((runtime,n_side,n_side))
    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth)
    ants = sugarscape.ants(vis, met, nants,w=w,theta=theta)
    fractions=np.zeros((runtime,nants))
    for i in range(runtime):
        #print(f'\r{i/runtime*100}% done',end='')
        x_ret[i]=x.copy()
        im_ret[i]=brd.currentsugar.copy()
        # get moves of all the living ants
        bestsug =ants.get_deterministic_sugar(board=brd.currentsugar,antpos=x,l=l)
        x[ants.alive] = ants.get_newpos(board=brd.currentsugar, antpos=x, l=l)
        fractions[i,ants.alive]=brd.currentsugar[tuple(x[ants.alive].T)]/bestsug

        # let living ants move and eat
        ants.eat(brd.currentsugar, x)
        # hunger
        ants.hunger(i)
        # grow sugar back
        brd.growsugar()
        if np.all(~ants.alive):
            break


    fitnesses=sugarscape.fitnessess(ants.lifetimes,runtime,ants.alive,ants.sugarlevels)
    fractions=np.nanmean(fractions,axis=0)
    fractions[fractions==0]=np.nan
    return fitnesses,fractions


def run_eval():
    runtime=1000
    directory = "files"
    files=os.listdir(directory)
    s=[f.split('_') for f in files]
    s = [x for x in s if x[0] != "run1"]
    runs=[int(f[1].replace("run","").strip()) for f in s]
    iters=[int(f[3]) for f in s]

    iters=np.array(iters)
    iters.sort()

    fitnesses=np.zeros((len(iters),nants))
    fractions=np.zeros((len(iters),nants))

    for i in range(len(iters)):
        print(f'\r{i/len(iters)*100}% done',end='')
        w=np.load(f'files/w_run1_iteration_{iters[i]}_.npy')
        theta=np.load(f'files/theta_run1_iteration_{iters[i]}_.npy')
        fitnesses[i],fractions[i]=runmodel(w,theta)

    np.save('files/run1_fitnesses',fitnesses)
    np.save('files/run1_fractions',fractions)


run_eval()


fitnesses=np.load('files/run1_fitnesses.npy')
fractions=np.load('files/run1_fractions.npy')

files=os.listdir('files')
s=[f.split('_') for f in files]
s=[x for x in s if x[0]!="run1"]
iters=[int(f[3]) for f in s]


iters=np.array(iters)
indices=np.argsort(iters)
iters=iters[indices]
#fitnesses=fitnesses[indices]
#fractions=fractions[indices]


print(fitnesses.shape)
print(fractions.shape)

meanfrac=np.mean(fractions,axis=1)
meanfit=np.mean(fitnesses,axis=1)



fig,axs=plt.subplots(1,2)

for i in range(70):
    axs[0].plot(iters,fractions[:,i],color='lightblue',alpha=0.05)
    axs[1].plot(iters,fitnesses[:,i],color='orange',alpha=0.05)

axs[0].plot(iters,meanfrac,color='k',label='mean fraction')
axs[1].plot(iters,meanfit,color='k',label='mean fitness')

axs[0].set_title('fractions')
axs[1].set_title('fitnesses')
axs[0].legend()
axs[1].legend()

axs[0].grid()
axs[1].grid()

plt.show()


