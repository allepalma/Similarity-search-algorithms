The file with the central algorithm is named main.py. It is pre-disposed to be launched from the terminal following these steps:

- cd the directory of the code
- you need to dispose of the dataset downloadable at https://drive.google.com/file/d/1Fqcyu9g6DZyYK_1qmjEgD1LlGD7Wfs5G/view?usp=sharing and 
  fix it in a repository at your discretion. 
- To launch the main.py executable, you need to provide the following syntax:

		python main.py -d /very/long/path/to/user_movie_rating.npy -s seed_of_interest -m similarity_measure

   As a seed, the user can type any random number. For the method, one needs to choose among js (Jaccard), cs (cosine) or dcs (discrete cosine).
 

The algorithm will print out the progress of the computation step by step and produce a file with the pairs of similar users it found in the 
following format:

u11,u12
u21,u22
.
.
.
un1,un2

where ui1<ui2 in terms of index. The name of the file generated will be js.txt, cs.txt or dcs.txt depending on the similarity measure provided. The file will be
produced in the same directory where the launching was performed. Any already existing file with the same name in the directory where the script is will 
be overwritten.

NB. The indexes of the users printed on the file start from 1 (not from 0 as in the canonical python indexing).

SUGGESTION:
- Work with a laptop of at least 8 GB of RAM, possibly plugged to a charger or without any power saving mode.
- If the terminal does not show the result after a while, try to press the spacebar
- Make sure to work with Python 64bit (32bit won't even read the input data)

Here are the specifics of the laptop used for the experiments:
Processor: Intel(R) Core(TM) i7-88550U CPU @ 1.80GHz 1.99GHz
RAM: 8,00 GB (7,89 utilizable)
Type of system: Operating system (Windows) 64bit 

