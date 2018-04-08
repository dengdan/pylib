import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import util
import sys
import numpy as np
image_dir =  util.io.get_absolute_path(sys.argv[1])
check_ok_path = util.io.get_absolute_path('~/data/good.txt')
invalid_path = util.io.get_absolute_path('~/data/invalid.txt')
unsure_path = util.io.get_absolute_path('~/data/not_sure.txt')
scene_invalid_path = util.io.get_absolute_path('~/data/scene.txt')
hard_examples_path = util.io.get_absolute_path('~/data/hard.txt')
def read_ids(path):
    if util.io.not_exists(path):
        return []
    return [util.str.remove_invisible(line) for line in util.io.read_lines(path)]
    

def add_id(path, id):
    util.io.make_parent_dir(path)
    with open(path, 'aw') as f:
        f.write('%s\n'%(id))
        
fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.2)
def get_id(name):
    return util.str.find_all(name, '[a-zA-Z0-9]{24}')[0]
    
class Index(object):
    def __init__(self):
        visited_ids = read_ids(check_ok_path) + read_ids(invalid_path) + \
                    read_ids(unsure_path) + read_ids(scene_invalid_path) + read_ids(hard_examples_path)
        image_names = util.io.ls(image_dir, '.jpg')
        self.to_be_visited = []
        for name in image_names:
            img_id = get_id(name)
            if img_id not in visited_ids:
                self.to_be_visited.append(name)
        self.idx = 0
        
    def get_image_name(self):
        return self.to_be_visited[self.idx]
    def get_image_id(self):
        name = self.get_image_name()
        return get_id(name)
    
    def show_image(self):
        image_path = util.io.join_path(image_dir, self.to_be_visited[self.idx])
        img = util.img.imread(image_path, rgb = True)
        ax.imshow(img)
        ax.set_title('%d/%d:%s'%(self.idx + 1, len(self.to_be_visited), self.get_image_name()))
        fig.canvas.draw()
    
    def check_ok(self, event):
        add_id(check_ok_path, self.get_image_id())
        self.next(event)
        
    def set_invalid(self, event):
        add_id(invalid_path, self.get_image_id())
        self.next(event)
        
    def set_hard(self, event):
        add_id(hard_examples_path, self.get_image_id())
        self.next(event)
        
    def set_unsure(self, event):
        add_id(unsure_path, self.get_image_id())
        self.next(event)
    def set_scene_invalid(self, event):
        add_id(scene_invalid_path, self.get_image_id())
        self.next(event)
        
    def next(self, event):
        self.idx += 1
        if self.idx >= len(self.to_be_visited):
            self.idx = len(self.to_be_visited) - 1
        self.show_image()

    def prev(self, event):
        self.idx -= 1
        self.show_image()

    def copy_id(self, event):
        cmd = 'echo %s | pbcopy'%(self.get_image_id())
        util.cmd.cmd(cmd)
        
callback = Index()
buttons = [
                    ('Previous', callback.prev),
                    ('Not sure', callback.set_unsure),
                    ('Wrong Ann', callback.set_invalid),
                    ('Check OK', callback.check_ok),
                    ('Scene Invalid', callback.set_scene_invalid),
                    ('Hard', callback.set_hard),
                    ('Next', callback.next), 
                    ('Copy ID', callback.copy_id), 
                    ]
pos = np.linspace(0, 1, len(buttons) + 1)
btns = []
for (txt, fn), x in zip(buttons, pos[:-1]):
    btn_ax = plt.axes([x, 0.05, 0.12, 0.075])
    btn  = Button(btn_ax, txt)
    btn.on_clicked(fn)
    btns.append(btn)

callback.show_image()
plt.show()