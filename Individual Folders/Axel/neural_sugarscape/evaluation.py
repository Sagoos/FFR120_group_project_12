import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sugarscape


def create_animation(ims, xpos):
    """
    Create an animation with a static background image and animated agent positions.

    Parameters:
    im (2D array): A n_side x n_side matrix for the background.
    xpos (3D array): A n_iterations x n_agents x 2 tensor containing agent positions.
    """

    print(xpos.shape)
    xpos=np.flip(xpos,axis=2)
    # Validate input
    n_iterations, n_agents, dims = xpos.shape
    if dims != 2:
        raise ValueError("Each agent position must have 2 dimensions (row, col).")

    n_side = ims.shape[1]
    if im.shape[1] != im.shape[2]:
        raise ValueError("The background image must be square.")

    # Create the figure and axis
    fig, ax = plt.subplots()
    image=ax.imshow(ims[0])
    scatter = ax.scatter([], [], c='red', s=50)

    # Set up axis limits and grid
    ax.set_xlim(-0.5, n_side - 0.5)
    ax.set_ylim(n_side - 0.5, -0.5)


    # Update function for the animation
    def update(frame):
        # Extract the positions of agents for this frame
        positions = xpos[frame]
        scatter.set_offsets(positions)
        image.set_data(ims[frame])
        return scatter,image

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=n_iterations, blit=True, interval=100
    )
    plt.show()
    return

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
runtime=2000
l=n_side-1




def runmodel(w,theta):
    x=np.random.randint(0,n_side,(nants,2))
    x_ret=np.zeros((runtime,nants,2))
    im_ret=np.zeros((runtime,n_side,n_side))
    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth)
    ants = sugarscape.ants(vis, met, nants,w=w,theta=theta)
    for i in range(runtime):
        print(f'\r{i/runtime*100}% done',end='')
        x_ret[i]=x.copy()
        im_ret[i]=brd.currentsugar.copy()
        # get moves of all the living ants

        x[ants.alive] = ants.get_newpos(board=brd.currentsugar, antpos=x, l=l)



        # let living ants move and eat
        ants.eat(brd.currentsugar, x)
        # hunger
        ants.hunger(i)
        # grow sugar back
        brd.growsugar()
        if np.all(~ants.alive):
            break

    return x_ret,im_ret

'''
w=np.load('files/w_run1_iteration_0_.npy')
tet=np.load('files/theta_run1_iteration_0_.npy')

x,im=runmodel(w,tet)
create_animation(im,x)
'''
w=np.load('files/w_run1_iteration_3998_.npy')
tet=np.load('files/theta_run1_iteration_3998_.npy')

x,im=runmodel(w,tet)
create_animation(im,x)

#w=np.tile(np.identity((vis*2+1)**2),(nants,1,1))
#tet=np.zeros((nants,(vis*2+1)**2))






