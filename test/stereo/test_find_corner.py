import util
from util.stereo import CalibrationBoard
import pdb

path = util.argv[1]
img = util.img.imread(path, mode = util.img.IMREAD_GRAY)
# img = img[420:580, 1600: 1800]

board = CalibrationBoard(n_rows = 6, n_cols = 8, square = 200)
ok, corners = board.find_corners(img, refine = False)

img = board.draw_corners(util.img.gray2bgr(img), corners = corners)
points = board.get_object_points(use_board_size = True)
# pdb.set_trace()
util.img.imshow("Img", img)
