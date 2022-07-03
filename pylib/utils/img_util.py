def compress_image(src:str, dest:str=None, target_size:float=30, unit:str = "KB")->int:
  """不改变图片尺寸压缩到指定大小
  Args:
      src (str): 压缩源文件
      dest (str, optional): 压缩文件保存地址. Defaults to None, 覆盖源文件
      target_size (float, optional): 目标大小, 单位KB. Defaults to 30. 受压缩比限制, 经常无法达到目标大小.
      unit (str, optional): 目标大小的单位. Defaults to KB

  Returns:
      int: 最终压缩大小
  
  https://wendao.blog.csdn.net/article/details/100579736?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2-100579736-blog-106544789.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2-100579736-blog-106544789.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=5
  """
  from utils.io_util import get_file_size, get_filename, mv
  import cv2
  if dest is None:
    dest = src
  size = get_file_size(src, unit)
  if size <= target_size:
    return size
  tmp_path = f'/tmp/{get_filename(src)}'
  im = cv2.imread(src, cv2.IMREAD_COLOR)
  is_jpg = src.endswith('.jpg')
  ratios = list(range(1, 10))[::-1]
  if is_jpg:
    ratios = list(range(1, 100, 5))[::-1]
  for ratio in ratios:
    if is_jpg:
      opt = cv2.IMWRITE_JPEG_QUALITY
      ratio *= 10
    else:
      opt = cv2.IMWRITE_PNG_COMPRESSION
    cv2.imwrite(tmp_path, im, [opt, ratio])
    size = get_file_size(tmp_path, unit)
    if size <= target_size:
      break
  mv(tmp_path, dest)
  return size