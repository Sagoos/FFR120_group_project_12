import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sugarscape
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize



def create_animation(ims_s,ims_d, xpos):
    """
    Create an animation with a static background image and animated agent positions.

    Parameters:
    im (2D array): A n_side x n_side matrix for the background.
    xpos (3D array): A n_iterations x n_agents x 2 tensor containing agent positions.
    """

    def create_transparent_colormap(base_cmap, alpha):
        base = plt.cm.get_cmap(base_cmap)
        new_cmap = base(np.arange(base.N))
        new_cmap[:, -1] = np.linspace(0, alpha, base.N)  # Vary alpha
        return ListedColormap(new_cmap)

    # Custom transparent colormap for danger
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
    img_d = ax.imshow(ims_d[0], cmap='Reds', norm=norm_d)
    scatter = ax.scatter([], [], c='red', s=50)

    # Set up axis limits and grid
    ax.set_xlim(-0.5, n_side - 0.5)
    ax.set_ylim(n_side - 0.5, -0.5)


    # Update function for the animation
    def update(frame):
        # Extract the positions of agents for this frame
        positions = xpos[frame]
        scatter.set_offsets(positions)
        img_d.set_data(ims_d[frame])
        img_s.set_data(ims_s[frame])
        return scatter,img_d,img_s

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=n_iterations, blit=True, interval=20
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
sugargrowth=0.03
#ant parameters-----------------------------------------
vis=2
met=3
nants=70

#GA parameters
runtime=2000
l=n_side-1

iterations_to_move=70



def runmodel(w,theta):
    iterations_to_kill=10
    spotpos = np.random.randint(0, n_side, (4, 2))

    x=np.random.randint(0,n_side,(nants,2))
    x_ret=np.zeros((runtime,nants,2))
    im_ret=np.zeros((runtime,n_side,n_side))
    dim_ret=np.zeros((runtime,n_side,n_side))

    brd=sugarscape.board(n_side,spotpos,spotwidths,maxsugar,sugargrowth,dangerpos,dangerwidth,maxdanger)
    ants = sugarscape.ants(vis, met, nants,w=w,theta=theta)
    sugers0=ants.sugarlevels
    for i in range(runtime):
        print(f'\r{i/runtime*100}% done',end='')

        if i%iterations_to_move==0:
            spotpos = np.random.randint(0, n_side, (4, 2))
            brd.update_sugarlevels(spotpos,spotwidths)
            brd.update_dangerlevels(np.random.randint(0,n_side,(3,2)),dangerwidth)

        x=x[ants.alive]
        ants=sugarscape.ants(vis,met,nants,w[ants.alive],tet[ants.alive])

        x_ret[i,ants.alive]=x.copy()
        x_ret[i,~ants.alive]=np.nan
        im_ret[i]=brd.currentsugar.copy()
        dim_ret[i]=brd.dangerlevels.copy()

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
    sugers=ants.sugarlevels
    return x_ret,im_ret,sugers,sugers0,dim_ret

'''
def Lorenz_curve(s):
    N = np.size(s)
    population_fraction = np.arange(N) / N
    s_sorted = np.sort(s)
    cumulative_s_fraction = np.cumsum(s_sorted) / np.sum(s_sorted)
    return cumulative_s_fraction, population_fraction

# L0, F0 = Lorenz_curve(S0)
# plt.figure(figsize=(5, 5))
# plt.plot(F0, L0, '.-', label='initial')
# plt.plot([0, 1], [0, 1], '--', color='k', label='equality')
# plt.legend()
# plt.xlabel('F')
# plt.ylabel('L')
# plt.title('sugar')
# plt.show()
'''



'''
w=np.load('files/w_run1_iteration_0_.npy')
tet=np.load('files/theta_run1_iteration_0_.npy')

x,im=runmodel(w,tet)
create_animation(im,x)
'''
w=np.load('files2/w_run1_iteration_21900_.npy')
tet=np.load('files2/theta_run1_iteration_21900_.npy')



x,im,sugers,sugers0,dangers=runmodel(w,tet)

'''
L0,F0=Lorenz_curve(sugers0)
L,F=Lorenz_curve(sugers)
plt.plot(L0,F0,label='Lorentz curve at start')
plt.plot(L,F,label='Lorentz curve at end')
plt.legend()
plt.show()
'''
create_animation(im,dangers,x)






