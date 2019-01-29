import util
from matplotlib import pyplot as plt
path = util.argv[1]

fig = plt.figure()
img = util.img.imread(path, rgb = True)

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.imshow(img)
plt.show()