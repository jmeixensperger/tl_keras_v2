import shutil
import os

old_dir = "/home/student/jmeixens/research_project/old/"
new_dir = "/home/student/jmeixens/research_project/new/"

def main():
    if os.path.isdir(old_dir) and os.path.isdir(new_dir):
        for d in os.listdir(old_dir):
            d_path = old_dir + d + "/"
            new_path = new_dir + d + "/"
            if os.path.isdir(d_path): 
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
