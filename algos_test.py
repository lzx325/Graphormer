import shutil
shutil.rmtree("lib.linux-x86_64-3.7",ignore_errors=True)
shutil.rmtree("temp.linux-x86_64-3.7",ignore_errors=True)
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()},build_dir=".")
import graphormer.algos as algos

adj_matrix=np.random.randint(0,2,(5,5))
M,path=algos.floyd_warshall(adj_matrix)
print(adj_matrix)
print(np.asarray(path))