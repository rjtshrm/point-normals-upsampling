# -*- coding: utf-8 -*-


def get_best_epoch(f_pointer):
    f_pointer.seek(0, 0) # begining of file
    read_states = f_pointer.readlines()
    read_min_loss = min(read_states, key=lambda k: float(k.split(", ")[1].split(" ")[1][0:-1]))
    best_epoch = int(read_min_loss.split(", ")[0].split(" ")[1])
    return best_epoch, read_min_loss



def get_current_state(f_pointer):
    f_pointer.seek(0, 0) # begining of file
    read_states = f_pointer.readlines()
    last_epoch = int(read_states[-1].split(", ")[0].split(" ")[1]) if len(read_states) else -1
    return last_epoch
