import shutil
import os

old_dir = "/home/sperry/Documents/multi-class/"
new_dir = "/home/sperry/Documents/multi-class2/"

def main():
    if os.path.isdir(old_dir) and os.path.isdir(new_dir):
        print("both directories are valid")
        for d in os.listdir(old_dir):
            d_path = old_dir + d + "/"
            new_path = new_dir + d + "/"
            if os.path.isdir(d_path):
                print("directory "+d+" is valid")
                for f in os.listdir(d_path):
                    true_name = f
                    if "_patient" in f:
                        pat_idx = f.find("_patient")
                        dot_idx = f.find(".jpg")
                        true_name = f[0:pat_idx] + f[dot_idx:] 
                    f_path = d_path + f
                    new_f_path = new_path + true_name
                    shutil.copyfile(f_path, new_f_path)

main()
