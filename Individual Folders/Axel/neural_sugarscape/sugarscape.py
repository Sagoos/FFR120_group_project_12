import numpy as np

class ants:
    def __init__(self,visibility,metabolisms,n_agents,w=None,theta=None,sugar0=30):
        '''
        :param visibility, visibility of ants:
        :param metabolism, metabolism of ants:
        :param n_agents, number of agents in population
        :param w, 3d tensor, each [i,:,:] slice is a weight matrix for an ant:
        :param theta, [i,:] is the threshholds for an ant:
        :param sugar0, starting suggar of each ant
        '''

        self.sugar0=sugar0

        self.vis=visibility
        self.met=metabolisms
        self.n_args=(2*visibility+1)**2
        self.n_ants=n_agents
        self.alive=np.ones(n_agents,dtype=bool)
        self.lifetimes=np.ones(n_agents)*np.nan
        #these will be used for getting the positions of each ant
        self.col_offsets = np.tile(np.arange(-self.vis, self.vis + 1), (2 * self.vis + 1, 1))
        self.row_offsets = np.tile(np.arange(-self.vis, self.vis + 1), (2 * self.vis + 1, 1)).T


        if w is None:
            self.w=np.random.randn(n_agents,(2*self.vis+1)**2,self.n_args)*np.sqrt(2/(self.n_args+2 * self.vis + 1))
        else:
            self.w=w
        if theta is None:
            self.theta = np.random.randn(n_agents,(2 * self.vis + 1) ** 2)*np.sqrt(2/(self.n_args+2 * self.vis + 1))
        else:
            self.theta = theta

        self.sugarlevels=np.ones(n_agents)*sugar0


    def reset(self):
        #resets the status of all ants
        self.alive[:]=True
        self.sugarlevels[:]=self.sugar0
        self.lifetimes[:]=np.nan

    def hunger(self,iteration):
        #ants burn through their sugar supply, die, and their lifetimes are recorded
        self.sugarlevels-=self.met
        self.lifetimes[self.sugarlevels<0&self.alive]=iteration
        self.alive[self.sugarlevels<0]=False

    def eat(self,board,antpos):
        #ants eat the sugar where they are
        self.sugarlevels[self.alive]+=board[tuple(antpos[self.alive].T)]
        board[tuple(antpos.T)]=0

    def get_newpos(self,board,antpos,l):
        #finding the positions the ants want to move to according to their neural networks
        rows = np.mod(self.row_offsets[None, :, :] + antpos[self.alive, 0, None, None], l)
        cols = np.mod(self.col_offsets[None, :, :] + antpos[self.alive, 1, None, None], l)
        options = board[rows, cols]
        options = options.reshape((options.shape[0], -1))  # flattens each matrix of options
        options=np.einsum('ijk,ik->ij', self.w[self.alive], options)-self.theta[self.alive]
        choices = np.array(np.unravel_index(np.argmax(options, axis=1), (2 * self.vis + 1, 2 * self.vis + 1))) - self.vis
        return np.mod(choices + antpos[self.alive].T, l).T

    def get_deterministic_newpos(self,board,antpos,l):
        #deterministic model for where the ants should move
        rows = np.mod(self.row_offsets[None, :, :] + antpos[self.alive, 0, None, None], l)
        cols = np.mod(self.col_offsets[None, :, :] + antpos[self.alive, 1, None, None], l)
        options = board[rows, cols]
        options = options.reshape((options.shape[0], -1))  # flattens each matrix of options
        #options = np.einsum('ijk,ik->ij', self.w[self.alive], options) -self.theta[self.alive]
        choices = np.array(np.unravel_index(np.argmax(options, axis=1), (2 * self.vis + 1, 2 * self.vis + 1))) - self.vis
        return np.mod(choices + antpos[self.alive].T, l).T


    def get_deterministic_sugar(self,board,antpos,l):
        # deterministic model for where the ants should move
        rows = np.mod(self.row_offsets[None, :, :] + antpos[self.alive, 0, None, None], l)
        cols = np.mod(self.col_offsets[None, :, :] + antpos[self.alive, 1, None, None], l)
        return np.max(board[rows, cols],axis=(1,2))

def crossover(arr1,arr2):
    #crossover of 2 arrrays
    N=arr1.size
    ind=np.random.randint(0,N)
    return np.concatenate((arr1.flatten()[0:ind],arr2.flatten()[ind:N])).reshape(arr1.shape),np.concatenate((arr2.flatten()[0:ind],arr1.flatten()[ind:N])).reshape(arr1.shape)

def mutate(arr,p_mut,sigma_mut):
    #mutation of select values in an array
    boolmask=np.random.choice([True,False],arr.shape,p=[p_mut,1-p_mut])
    arr[boolmask]+=np.random.randn(np.sum(boolmask))*sigma_mut

def roullette_wheel(fitnesses):
    #returns indices of winning individuals
    return np.random.choice(fitnesses.size,fitnesses.size,p=fitnesses/np.sum(fitnesses))

def fitnessess(lifetimes,maxlife, living, sugarlevel):
    #calculates fitnesses of ants at the end of a run
    fit=sugarlevel.copy()
    fit-=np.min(fit)
    fit[~living]*=(lifetimes[~living]/maxlife)
    return fit


class board:
    def __init__(self,n_side,spotpos,spotwidths,max_sugar,sugargrowth):
        '''
        :param n_side: sidelenths of board
        :param spotpos: the positions of the circular zones of high sugar, Nxn
        :param spotwidths: the widths of each zone
        :param max_sugar: the highest level of sugar allowed on the board
        :param sugargrowth: the fraction of max sugar for a given cell to be regrown each update
        '''

        def gauss(r,sigma):
            return np.exp(-r**2/sigma**2)

        R=np.zeros((len(spotpos),n_side,n_side))
        x=np.arange(n_side)
        y=np.arange(n_side)
        X,Y=np.meshgrid(x,y)



        for i in range(len(spotpos)):
            dx=(X-spotpos[i,0]).astype(np.float64)
            dy=(Y-spotpos[i,1]).astype(np.float64)
            dx-=np.round(dx/n_side)*n_side
            dy-=np.round(dy/n_side)*n_side
            R[i]=np.sqrt((dx)**2+(dy)**2)
        sugs=np.sum(gauss(R,spotwidths[:,np.newaxis,np.newaxis]),axis=0)
        sugs*=max_sugar/np.max(sugs)
        self.sugarlevels=sugs.copy()
        self.currentsugar=sugs.copy()
        self.sugargrowth=sugargrowth


    def growsugar(self):
        #replenishes sugar
        self.currentsugar+=self.sugarlevels*self.sugargrowth
        bools=self.currentsugar>self.sugarlevels
        self.currentsugar[bools]=self.sugarlevels[bools]

    def reset(self):
        #resets sugarlevels to baseline
        self.currentsugar=self.sugarlevels.copy()

