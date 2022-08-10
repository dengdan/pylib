def main(paths, target_size, unit):
  import utils.img_util, utils.io_util
  for path in paths:
    origin_size = utils.io_util.get_file_size(path, unit=unit)
    out_size = utils.img_util.compress_image(src = path, target_size=target_size, unit=unit)
    print(f"{path}: {origin_size:.2f}{unit} --> {out_size:.2f}{unit}")

if __name__ == "__main__":
  import argparse
  import sys
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-f', '--file', nargs='+', required=True, help="image paths")
  parser.add_argument('-s', '--target-size', type=float, default=30,
            help="target size")
  parser.add_argument('-u', '--unit', type=str, default='KB', help="size unit")
  args = parser.parse_args()
  paths = args.file
  if not paths:
    paths = sys.argv[1:]
  main(paths, args.target_size, args.unit)