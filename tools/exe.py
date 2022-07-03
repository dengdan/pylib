
def run(name, *kargs, **kwargs):
  from importlib import import_module
  data = name.split('.')
  import utils
  assert len(data) == 2, "the name must be in the format of module_name.fn_name"
  import_module(f"utils.{data[0]}")
  fn = eval(f"utils.{name}")
  print(fn(*kargs, **kwargs))
  
if __name__ == "__main__":
  import fire
  fire.Fire(run)