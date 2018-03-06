import os
import shutil

file2change = "/home/student/jmeixens/research_project/tl_keras_v2/research/patient-groups.csv"
datnew = "/home/student/jmeixens/research_project/tl_keras_v2/research/pat-groups.csv"


def main():

	outs = open(datnew, 'w')
	ins = open(file2change, 'r')
	for line in ins:	
		pat_index = line.find('/')
                pat_dir = line[0:pat_index]
                new_line = ''
                if pat_dir == 'diseased':
                    if 'emphysema' in line:
                        new_line = 'emphysema/' + line[pat_index+1:] 
                    elif 'fibrosis' in line:
                        new_line = 'fibrosis/' + line[pat_index+1:] 
                    elif 'micronodules/' in line:
                        new_line = 'micronodules/' + line[pat_index+1:] 
                    elif 'ground_glass/' in line:
                        new_line = 'ground_glass/' + line[pat_index+1:] 
		else:
                    new_line = line
                outs.write(new_line)
main()
