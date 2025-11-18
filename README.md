# FLARE: An Adaptive Reputation for Dynamic Client Reliability Assessment in Federated Learning
\[...\]
# Experiments
Requirements:
| Dependency             | Version        |
|------------------------|----------------|
| flwr                   | ==1.20.0       |
| flwr[simulation]       | ==1.20.0       |
| flwr-datasets[vision]  | >=0.5.0        |
| pandas                 | ==2.3.1        |
| scipy                  | ~=1.16.1       |
| torch                  | ==2.8.0        |
| torchvision            | ==0.23.0       |
| numpy                  | ~=2.3.2        |
| scikit-learn           | ~=1.7.1        |

> requirements.txt together with pyproject.toml includes all of these and are downloaded in the serverapp docker image

## SLURM managed Cluster

We performed experiments on a High Performance Cluster utilizing 13 Nodes for 100 clients.

Each Node is constructed as shown in the following table:
| Component             | Specification                             |
|-----------------------|-------------------------------------------|
| CPU                   | Intel Xeon Gold 6248R (24 cores @ 3.0GHz) |
| GPU                   | 4× NVIDIA RTX 2070 (8GB each)             |
| RAM                   | 256GB DDR4 ECC                            |
| Storage               | 2TB NVMe SSD                              |
| Operating System      | Ubuntu 24.04 LTS                          |

The following sections show execution of the experiments on that cluster and can be applied to any SLURM managed cluster.

### Setup

To run the experiments on a slurm based cluster we provided some bash scripts (found in /scripts) to make the deployment easier.
Use scp to get the scripts onto your cluster, where you want to run the experiments. Also copy the pyproject.toml over to the cluster as this will allow you to configure hyperparameters and scenarios.
The scripts rely on Singularity so make sure it is available on your cluster.

So first you must build the images, we used docker for this but you can write your own Singularity image as well.
Build the docker images for the platform that you need, upload them to docker hub to be able to pull them on any machine.

After that use Singularity to pull the required docker images to run our framework.
```bash
singularity pull docker://<your-image>:latest
```

Furthermore, make sure to create a folder called `/results` as it is required by the superlink container to start and write results into.

### Start Flower Infrastructure

First of all, change any host paths for volume binds in .slurm and .sh scripts to match with your host paths. Note: the host path will not be automatically created during the experiments runtime and must exist before starting any containers.

For instance: ./superlink.sh has a line to bind result paths from host and container. Change these paths to your needs. You might have to check out what path results are written to in your superlink container as this depends on what program you use to run images.

To prepare the experiment setting you need to get Flowers Infrastructure running.
We provide the script `superlink.sh` to start Flowers Superlink. Output will be written to `./superlink.out`. Check this file as it prints the superlinks IP adress, which is required for the next step.

To start Flowers Supernodes (which will act as clients) you can use our provided script `clients.sh`. However, before running you might want to specify the number of clients to use in your Federation. Do this by modifying `clients.slurm` here you can modify the lines `#SBATCH --ntasks=100` and `--node-config "partition-id=$SLURM_PROCID num-partitions=100"'` to change 100 to your needs. Once this is set up run `./clients.sh <Superlink-IP>` to start the clients. You can use `cat clients.out` to check when they are ready for the experiment. This can take around half a minute for all clients to start.

### Start the Remote Federation

After the clients are running the last thing you have to do is to start the remote federation, do this by first configuring the `pyproject.yaml` to your needs (The main part you want to modify is \[tool.flwr.app.config\]) where you can configure whether to use non-iid or not, what dataset to use, what attack pattern to use etc.

VERY IMPORTANT: The first experiment for a given dataset must run with one client only (and recommended to use only one server round and limit dataset size to a low number like 100, configure all mentioned settings in pyproject.toml). The dataset will be downloaded to your host environment and will be reused for further runs. This is needed to avoid download conflicts. After this has completed you can continue with your experiments for the dataset you just downloaded.

Once the `pyproject.toml` is configured you can run `./server.sh`. The output of the app and client classification as well as their current scores can be checked in `byebye-badclients.out`.

### Results

You get results in the /results directory you created earlier. You will find a config.json that contains your runtime configuration. You will find 3 folders of measurements `latency`, `performance` and `robustness`. In the `latency` folder you will see the aggregation time (as this is the most relevant and time consuming part of our framework). The `performance` folder contains measurements of accuracy, loss, f1-, recall-, and precision-scores per round. And the `robustness` folder contains robustness measurements with global scope (`/overall`) and an isolated per-round scope (`/per_round`) from various perspectives.
In `/overall` or `/per_round` you will find `robustness_score.csv` which shows the very general decision making accuracy per round. The best score is 1 which means the framework Untrusted all Malicious clients and Trusted all Benign clients. You get more insight in how many Malicious clients were untrusted by looking at `hard_malicious_detection_rate.csv` or `soft_malicious_detection_rate` which is a better measurement for our framework as we not only fully exclude or fully include clients from aggregation but also partially include clients for aggregation, the `soft` measurements do exactly that, they measure how much impact did the client really get on aggregation, while the `hard` measurements show only full in- or exclusion.
