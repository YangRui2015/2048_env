import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time


def log2_shaping(s, divide=16):
    s = np.log2(1 + s) / divide
    return s


def check_path_exist(path, verbose=True):
    if not os.path.exists(path):
        os.mkdir(path)
        if verbose:
            print("make the dir {} finished".format(path))
    else:
        if verbose:
            print("the directory {} already exists".format(path))

def running_average(lis, length=5):
    if len(lis) > 10:
        end = len(lis) // length
        lis = lis[:end * length]
        arr = np.array(lis).reshape(-1, length)
        arr = arr.mean(axis=1)

        return list(arr.reshape(-1))
    else:
        return lis
    
 
def plot_save(lis, path, title=None, x_label=None, y_label=None):
    dir = path.split("/")[:-1]
    dir = "/".join(dir) + "/"
    check_path_exist(dir, verbose=False)
    plt.figure()
    if type(lis[0]) == list:
        for li in lis:
            plt.plot(li)
    else:
        plt.plot(lis)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.savefig(path)
    plt.close("all")


def del_dir_tree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except:
            print("remove path {} failed!".format(path))


def del_files(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path, file))
        print("Remove files in {}".format(path))
    elif os.path.isfile(path):
        os.remove(path)
        print("Remove file {}".format(path))
    else:
        print("{} not a file or a directory".format(path))



class Perfomance_Saver():
    '''目前先支持txt'''
    def __init__(self, path='performance_data.txt'):
        self.path = path
        self.clear_file()

    def clear_file(self):
        with open(self.path, 'w') as file:
            file.write('clear since :{}\n\n'.format(time.ctime()))
        print("clear file finished")

    def save(self, performance_list, info):
        with open(self.path, 'a+') as file:
            file.writelines("time: {}\n".format(time.ctime()))
            file.writelines("info: {} \n".format(str(info)))
            performance_str = [str(x) + " " for x in performance_list]
            file.writelines(performance_str)
            file.writelines('\n\n')
        print('write to file finished')


class Model_Saver():
    '''存一定数量高分模型，防止模型存过多'''
    def __init__(self, num=10):
        self.num_max = num
        self.path_list = []
    
    def save(self, path):
        if len(self.path_list) >= self.num_max:
            os.remove(self.path_list.pop(0))
            print('del surplus modle files')

        self.path_list.append(path)
        


