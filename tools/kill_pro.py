
def kill(name):
    from utils.proc_util import pkill, ps_aux_grep, get_pid
    import utils.str_util
    lines = utils.str_util.split(ps_aux_grep(name), '\n')
    excludes = ["kill_pro.py", "grep "]
    pids = get_pid(name, excludes=excludes)
    print(pids)
    for line in lines:
        for pid in pids:
            if f" {pid} " in line:
                print(line)
    yes = input('kill them all?[n] y/n.')
    if yes == 'yes' or yes == 'y' or yes == 'Y':
        pkill(pids)
    else:
        try:
            yes = int(yes)
            pkill(yes)
        except:
            pass

if __name__ == "__main__":
    import fire    
    fire.Fire(kill)
