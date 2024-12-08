sugarscape.py is a class with the objects ants and board.
ants is a population of ants with relevant information and functions such as getting the positions of the ants in the next iteration and letting the ants eat and die.
board is the board on which the ants move. It stores sugarlevels and the dangerlevels of "natural catastrophes".

main.py is the main class for training the mini neural networks in the ants.

evaluation.py makes an animation of the ants moving around on a board. It can be used to look at how they are performing.

plot_evolution.py has functions for plotting the ants fitness and some other performance related statistics as a function of training iteration.

moving_sugar.py is similar to evaluation.py except the sugar and dangerzones change positions every now and then.

files has training and evaluation files from a training run that was conducted with a bug that gave the ants a bad fitness function

files2 is a training run when the bug was fixed, for the training iterations after 22000 in this file i changed the mutation parameters and it made the ants a lot worse,

files3 is files2 but with data from after iteration 21900 deleted. This is the home of the finest ants in this directory, the discerning gentlemans choice.