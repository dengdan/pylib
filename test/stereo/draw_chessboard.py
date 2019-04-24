#encoding=utf-8
import util
from util.stereo import CalibrationBoard
import pdb
CELL_IN_MM = 200
PIXELS_PER_MM = int(util.argv[1])
COLS = 5
ROWS = 4
CROP = 0

WHITE = 255
BLACK = 0
START_COLOR = BLACK
cell_width = CELL_IN_MM * PIXELS_PER_MM

origin_width = cell_width * COLS
origin_height = cell_width * ROWS

img = util.img.black((origin_height, origin_width))

# pdb.set_trace()
pre_row_start_color = 255 - START_COLOR
for row in range(ROWS):
    row_start_color = 255 - pre_row_start_color
    pre_row_start_color = row_start_color
    pre_cell_color = 255 - row_start_color
    for col in range(COLS):
        current_cell_color = 255 - pre_cell_color
        pre_cell_color = current_cell_color
#         if current_cell_color == WHITE:
#             continue
        rect_lt = (col * cell_width, row * cell_width)
        rect_rb = ((col + 1) * cell_width - 1, (row + 1) * cell_width - 1)
        util.img.rectangle(img, rect_lt, rect_rb, color = current_cell_color, border_width = -1)

if CROP:
    crop = int(cell_width * CROP)
    left_border = cell_width - crop
    radius = left_border // 3
    img = img[crop : -crop, crop : - crop]
h, w = img.shape
# crop corners
# col_corners = [0, 2 * cell_width - crop]
# while len(col_corners) < COLS:
#     col_corners.append(col_corners[-1] + 2 * cell_width)

# for corner in col_corners:
#     lu = [corner, 0]
#     rb = [v + radius for v in lu]
#     util.img.rectangle(img, lu, rb, color = 255 - START_COLOR, border_width = -1)
#     util.img.circle(img, rb, r = radius, color = START_COLOR, border_width = -1)
#     
#     lu = [w - corner - radius, 0]
#     rb = [v + radius for v in lu]
#     util.img.rectangle(img, lu, rb, color = 255 - START_COLOR, border_width = -1)
#     center = [lu[0], lu[1] + radius]
#     util.img.circle(img, center, r = radius, color = START_COLOR, border_width = -1)
    

board_width = (COLS - CROP * 2) * CELL_IN_MM
board_height = (ROWS - CROP * 2) * CELL_IN_MM
print("board_width = %d mm, board_height = %d mm"%(board_width, board_height))

print(util.img.imwrite(img = img, path = "~/temp/no-use/images/方格边长%dmm-有效区域%dmmx%dmm-%d.jpg"\
                            %(CELL_IN_MM, board_width, board_height, PIXELS_PER_MM)))
# util.cit(img, name = "%d_pixels_per_mm_"%(PIXELS_PER_MM))

# board = CalibrationBoard(n_rows = ROWS - 1, n_cols = COLS - 1, square = 200)
# ok, corners = board.find_corners(img)
# if not ok:
#     print("Invalid chessboard")
#     exit(0)
# img = board.draw_corners(util.img.gray2bgr(img), corners = corners)
# points = board.get_object_points(use_board_size = True)
# pdb.set_trace()
# util.img.imshow("Img", img)

# util.img.imshow("board", img)
