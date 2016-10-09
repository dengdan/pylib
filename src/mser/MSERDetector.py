#coding=utf-8
'''
Created on 2016年10月5日

@author: dengdan
'''
import numpy as np
class Point(object):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class ER(object):
    def __init__(self, root, points):
        self.children = None 
        self.root = root
        self.points = points
        self.parent = None
        
    def add_child(self, child):
        if self.children == None:
            self.children = []
        self.children.append(child)
    
    def set_parent(self, parent):
        assert self.parent == None, "Error when doing union"
        self.parent = parent
    def get_area(self):
        return len(self.points)
    
    def get_unstableness(self, child):
#         if self.parent == None or self.children == None:
#             return np.infty   
        return (self.get_area() - child.get_area()) * .1 / child.get_area()
     
def from_image(image):
    points = [(-1, -1)] * np.prod(image.shape)  
    rows, cols = image.shape
    for row in xrange(rows):
        for col in xrange(cols):
            points[cols * row + col] = Point(image[row, col], (row, col))
            
    return points
    
def detect(image, threshold_delta = 2, max_area_variation = 0.25, region_area_range = [30, 14000] ):
    rows, cols = image.shape
    points = from_image(image)
    idx = [None] * (rows * cols)
    status = []
    def find_parent(pos):
        row, col = pos
        loc = row * cols + col
        return idx[loc]
    
    def find_root(pos):
        parent = find_parent(pos)
        if parent == None:
            return None
        elif parent == pos:
            return pos
        else:
            return find_root(parent)
    
    def set_flag(pos, flag):
        row, col = pos
        loc = row * cols + col
        idx[loc] = flag
        
    def set_root(pos):
        set_flag(pos, pos)
        
    def union(r1, r2):
        root1 = find_root(r1)
        root2 = find_root(r2)
        if root1 == None or root2 == None:
            return

        if root1 == root2:
            return
        
        v1 = image[root1[0], root1[1]]
        v2 = image[root2[0], root2[1]]
        if v1 < v2:
            set_flag(root2, root1)
        else:
            set_flag(root1, root2)
    
    def find_neighbours(pos):
        row, col = pos
        pre_neighbours = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
        neighbours = []
        for p in pre_neighbours:
            if p[0] < rows and p[1] < cols:
                neighbours.append(p)
        return neighbours
    
    
    def find_ERs():
        regions = {}
        rows, cols = image.shape
        for row in xrange(rows):
            for col in xrange(cols):
                root = find_root((row, col))      
                if root == None:
                    continue
                if  root not in regions:
                    regions[root] = []
                regions[root].append((row, col))
                
        for key in regions:
            regions[key] = ER(key, regions[key])
        return regions
    
    
    def gen_tree():
        assert len(status[-1]) == 1, 'There should be only one ER at last'
        last_status = status[-1]
        parents = last_status
        for children in status[-2::-1]:
            children_roots = children.keys()
            for children_root in children_roots:
                for p in parents.keys():
                    if children_root in parents[p].points:
                        parents[p].add_child(children[children_root])
                        children[children_root].set_parent(parents[p])    
            parents = children
                            
        
        root_ER = last_status[last_status.keys()[0]]
        return root_ER
    
    print 'sorting...'
    points.sort(lambda p1, p2: int(p1.value) - int(p2.value))
    print 'sort ended...'
    
    thresholds = []
    threshold = 0
    while threshold < 255:
        thresholds.append(threshold)
        threshold += threshold_delta
    
    thresholds.append(255)
    print 'thresholds:', thresholds
    for threshold in thresholds:
        while len(points) > 0 and points[0].value <= threshold:
            point = points.pop(0)
            neighbours = find_neighbours(point.pos)
            set_root(point.pos)
            for neighbour in neighbours:
                union(point.pos, neighbour)
        
        current_ERs = find_ERs()
        print 'threshold = %d,'%(threshold), 'Extremal Regions found: %d '%(len(current_ERs))
        status.append(current_ERs)
        
    print 'generating tree...'
    root_ER = gen_tree()
    
    ercount = 0
    for s in status:
        ercount += len(s)
    print "Total ERs:", ercount 
    queue = [root_ER]
    leaves = []
    
    print 'finding Maximum Stable Extremal Regions...'
    while len(queue) > 0:
        er = queue.pop()
        if er.children == None:
            leaves.append(er)
        else:
            queue.extend(er.children)
    msers = set([])
    print "Leaves: ",len(leaves)
    for leaf in leaves:
        node = leaf.parent
        child = leaf
        while node != None:
            unstableness = node.get_unstableness(child)
            if unstableness < max_area_variation:
                if node.get_area() >= region_area_range[0] and node.get_area() <= region_area_range[1]:  
                    msers.add(node)
            child = node
            node = node.parent
    return msers

import cv2
img_org =  cv2.imread('../demo/data/cat.jpg')
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
msers = detect(img)
print len(msers)