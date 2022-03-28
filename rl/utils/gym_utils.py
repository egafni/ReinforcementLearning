import torch
from IPython import display
from matplotlib import animation, pyplot
from pyvirtualdisplay import Display


def jupyter_animation(env, n_steps=100, net=None, o=None):
    with Display():
        frames = []
        for _ in range(n_steps):
            if net is None:
                # Randomly sample an action
                a = env.action_space.sample()
            else:
                device = "cuda" if next(net.parameters()).is_cuda else "cpu"
                # use the NN to select an action
                assert o is not None, "if passing a net, must pass o, the current observation"
                o = torch.Tensor(o).unsqueeze(0).to(device)  # add batch dim
                pi = net(o)
                a = int(torch.argmax(pi))

            o, r, d, i = env.step(a)  # Take action from DNN in actual training.

            # display.clear_output(wait=True)
            frames.append(env.render("rgb_array"))

            if d:
                env.reset()

        animate(frames)


def animate(frames, dpi=72, interval=50):
    pyplot.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = pyplot.imshow(frames[0])
    pyplot.axis = "off"

    def animate(i):
        return patch.set_data(frames[i])

    ani = animation.FuncAnimation(pyplot.gcf(), animate, frames=len(frames), interval=interval)
    display.display(display.HTML(ani.to_jshtml()))
