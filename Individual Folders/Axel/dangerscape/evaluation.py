import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sugarscape
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap




def create_animation(ims_s,ims_d, xpos,living):
    """
    Create an animation with animated agent positions and background sugar.

    Parameters:
    ims_s (3D array): A n_iteration x n_side x n_side matrix for the background.
    ims_d (2d array): A n_side x n_side matrix for the background dangerlevels.
    xpos (3D array): A n_iterations x n_agents x 2 tensor containing agent positions.
    """

    def create_transparent_colormap(base_cmap, alpha):
        base = plt.cm.get_cmap(base_cmap)
        new_cmap = base(np.arange(base.N))
        new_cmap[:, -1] = np.linspace(0, alpha, base.N)  # Vary alpha
        return ListedColormap(new_cmap)

    # custom transparent colormap for danger
    transparent_blues = create_transparent_colormap('Blues',alpha=1)



    norm_s = Normalize(vmin=0, vmax=maxsugar)
    norm_d = Normalize(vmin=0, vmax=maxdanger)
    xpos=np.flip(xpos,axis=2)
    # Validate input
    n_iterations, n_agents, dims = xpos.shape
    if dims != 2:
        raise ValueError("Each agent position must have 2 dimensions (row, col).")

    n_side = ims_s.shape[1]

    # Create the figure and axis
    fig, ax = plt.subplots()
    img_s = ax.imshow(ims_s[0], cmap=transparent_blues, norm=norm_s)
    img_d = ax.imshow(ims_d, cmap='Reds', norm=norm_d)
    scatter = ax.scatter([], [], c='red', s=50)

    # Set up axis limits and grid
    ax.set_xlim(-0.5, n_side - 0.5)
    ax.set_ylim(n_side - 0.5, -0.5)


    # Update function for the animation
    def update(frame):
        # Extract the positions of agents for this frame
        positions = xpos[frame,lives[frame]]
        scatter.set_offsets(positions)
        # update sugarlevels
        img_s.set_data(ims_s[frame])
        return scatter,img_s

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=n_iterations, blit=True, interval=10
    )
    plt.show()
    return

#board parameters
n_side=200
spotpos = np.random.randint(0, n_side, (4, 2))
spotwidths = np.ones(len(spotpos)) * 40

dangerpos=np.random.randint(0,n_side,(3,2))
dangerwidth=np.ones(len(dangerpos))*20
maxdanger=20

maxsugar=10
sugargrowth=0.004
#ant parameters-----------------------------------------
vis=2
met=3
nants=70

#misc parameters
runtime=2000
l=n_side-1




def runmodel(w,theta):
    x=np.random.randint(0,n_side,(nants,2))
    x_ret=np.zeros((runtime,nants,2))
    im_ret=np.zeros((runtime,n_side,n_side))
    life_ret=np.ones((runtime,nants),dtype=bool)

    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth,dangerpos,dangerwidth,maxdanger)
    ants = sugarscape.ants(vis, met, nants,w=w,theta=theta)
    for i in range(runtime):
        print(f'\r{i/runtime*100}% done',end='')
        x_ret[i]=x.copy()
        im_ret[i]=brd.currentsugar.copy()
        life_ret[i]=ants.alive
        # get moves of all the living ants
        x[ants.alive] = ants.get_newpos(board=brd.currentsugar,dboard=brd.dangerlevels, antpos=x, l=l)
        # let living ants move and eat
        ants.eat(brd.currentsugar, x)
        # hunger
        ants.hunger(i,brd.dangerlevels,x)
        # grow sugar back
        brd.growsugar()
        if np.all(~ants.alive):
            break

    im_ret_d=brd.dangerlevels

    return x_ret,im_ret,im_ret_d,life_ret


#loading network parameters
w=np.load('files3/w_run1_iteration_21900_.npy')
tet=np.load('files3/theta_run1_iteration_21900_.npy')


#this makes a neural network that is equivalent to a deterministic ant that really doesn't want to go in the danger
'''
w=np.zeros(((vis*2+1)**2,2*(vis*2+1)**2))
arr=np.array([*np.ones((vis*2+1)**2),*(np.ones((vis*2+1)**2)*(-10))])
np.fill_diagonal(w[:,:25],arr[:25])
np.fill_diagonal(w[:,25:],arr[25:])
tet=np.zeros((2*vis+1)**2)
w=np.tile(w,(nants,1,1))
tet=np.tile(tet,(nants,1))
'''

#running model and making animation
x,im,imd,lives=runmodel(w,tet)
create_animation(im,imd,x,lives)





