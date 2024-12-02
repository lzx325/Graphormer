set -e
name="graphormer"
time="48"
n_cpus="32"
memory="100"
mkdir -p "slurm"
sbatch <<- EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -J ${name}
#SBATCH --partition=batch
#SBATCH -o slurm/%J.out
#SBATCH -e slurm/%J.err
#SBATCH --time=${time}:00:00
#SBATCH --mem=${memory}G
#SBATCH --cpus-per-task=${n_cpus}
#SBATCH --gres=gpu:4
#SBATCH --constraint="[v100]"
#SBATCH --account conf-aaai-2021.09.09-gaox
#run the application:
bash ogbg-molhiv.sh
EOF

