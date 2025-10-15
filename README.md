<h1 align="center">Graph‑RL Autoscaler: GNN‑Driven Agent for Cluster‑Based Microservice Scaling</h1>

<div align="center">
  <a href="https://github.com/SlyPex/graph-gym-hpa/actions"><img src="https://img.shields.io/github/actions/workflow/status/SlyPex/graph-gym-hpa/ci.yml?event=pull_request&style=plastic&logo=github&logoSize=auto&label=Continuous%20Integration&labelColor=181717" alt="Status Checks"/></a>
  <a href="https://github.com/SlyPex/graph-gym-hpa/commits/"><img src="https://img.shields.io/github/commit-activity/t/SlyPex/graph-gym-hpa?style=plastic&logo=github&logoSize=auto&labelColor=181717&color=white" alt="GitHub commit activity"/></a>
  <a href="https://github.com/SlyPex/graph-gym-hpa/README.md#collaborators"><img src="https://img.shields.io/badge/Collaborators-3-black?style=plastic&logo=github&labelColor=181717&color=white" alt="Collaborators"/></a>
  <br/>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-v3.11.13-yellow?style=plastic&logo=python&logoColor=white&logoSize=auto&labelColor=3776AB&color=ffd94a"/></a>
  <a href="https://kubernetes.io/"><img src="https://img.shields.io/badge/Kubernetes-v1.33.3-grey?style=plastic&logo=kubernetes&labelColor=white&color=%23326CE5"/></a>
  <a href="https://locust.io/"><img src="https://img.shields.io/badge/Locust-v2.10.2-blue?style=plastic&logo=locust&labelColor=white&color=b8ee4b"/></a>
  <br/>
  <a href="https://prometheus.io/"><img src="https://img.shields.io/badge/Prometheus-white?style=plastic&logo=prometheus&logoSize=auto&labelColor=white&color=E6522C"/></a>
  <a href="https://istio.io/"><img src="https://img.shields.io/badge/istio-v1.26.2-white?style=plastic&logo=istio&logoSize=auto&labelColor=white&color=466BB0"/></a>

</div>

## Description 📝

### Introduction
Current autoscalers (e.g., the Kubernetes HPA) are mostly reactive and do not account for inter-service dependencies. To address this limitation, this project combines two complementary ideas:

- Reinforcement Learning: allows an agent to learn by interacting with a real-world environment, mainly a Kubernetes cluster.
- Graph Neural Networks: because microservices and their call relationships form a graph, GNNs are well suited to model and learn inter-service dependencies.

### Cluster Stack 🏗️
<div align="center">
    <img width="736" height="481" alt="training_loop" src="https://github.com/user-attachments/assets/669c303a-cb14-446f-bc03-214176d21e9e" />
    <p>Figure: Shows the training loop of the agent and its interaction with the environment</p>
</div>


<div align="center">
  
|Tool|Utility
|:-:|:-:|
| <a href="https://kubernetes.io/"><img src="https://img.shields.io/badge/Kubernetes-white?style=plastic&logo=kubernetes&labelColor=white&color=%23326CE5"/></a> | Container Orchestration |
| <a href="https://locust.io/"><img src="https://img.shields.io/badge/Locust-white?style=plastic&logo=locust&color=b8ee4b"/></a> | User Load Simulation |
| <a href="https://github.com/GoogleCloudPlatform/microservices-demo/tree/main"><img src="https://img.shields.io/badge/Online_Boutique-white?style=plastic"/></a> | Benchmark Application |
| <a href="https://istio.io/"><img src="https://img.shields.io/badge/istio-white?style=plastic&logo=istio&logoSize=auto&labelColor=white&color=466BB0"/></a> | Injects a proxy at the pod level to monitor traffic between services |
| <a href="https://prometheus.io/"><img src="https://img.shields.io/badge/Prometheus-white?style=plastic&logo=prometheus&logoSize=auto&labelColor=white&color=E6522C"/></a>| Gathers metrics from different deployed tools (istio, locust, onlineboutique) |

</div>

### Project Files 📂
```
├── datasets
│   └── online_boutique_gym_observation.csv
├── k8s_config_files
│   ├── locust_files
│   │   ├── Dockerfile
│   │   ├── locustfile.py
│   │   └── locust.yaml
│   ├── onlineboutique.yaml
│   └── prometheus.yaml
├── requirements.txt
├── setup.py
└── src
  ├── gym_hpa
  │   ├── gnn
  │   │   ├── gnn.py
  │   │   └── graphCreation.py
  │   ├── paths.py
  │   └── rl_environments
  │       ├── deployment.py
  │       ├── online_boutique.py
  │       └── util.py
  └── policies
      ├── run
      │   └── run.py
      └── util
            └── util.py
```
- `datasets` : A directory contains the dataset which is used to train the agent in simulation mode.
- `k8s_config_files` : A configuration files directory in order to properly deploy the cluster stack.
- `requirements.txt` : A file which holds the required packages in order to run this project.
- `setup.py` : A Python script used to describe a package (metadata, dependencies, packaging instructions) in our case, the local package ![gym-hpa](./src/gym-hpa/).
- `src` : Holds the whole codebase of the developed framework.

  - `gym-hpa`:
    - `gnn`: the ![graphCreation](./src/gym-hpa/gnn/graphCreation.py) and the ![gnn](./src/gym-hpa/gnn/gnn.py) code is under this directory.
    - `rl_environments`: Holds the logic of handling the ![OnlineBoutique](./src/gym-hpa/rl_environments/online_boutique.py) app and its interaction with the K8s cluster.

  - `policies/run` : The main start point of this project to start a training/testing of an agent.

## Usage 🚀
The following steps require files from this repository. First, clone the project:
  - ```bash
    git clone https://github.com/SlyPex/graph-gym-hpa.git
    ```

### Cluster Setup 🛠️

Our cluster consists of the following VMs (Nodes):

- One Masternode (10 CPU Cores, 10 GB of RAM)
- Two Workers (8 CPU Cores, 8 GB of RAM)

> [!IMPORTANT]
> The following steps assume an existing Kubernetes cluster has already been set up, the previous specs and the setup method (kubeadm, Minikube, kind, etc.) is irrelevant, just make sure that the given resources (CPU Cores, RAM) to the cluster are more than enough to handle the previous stack.

1. Start by installing ![istio](https://istio.io) using `istioctl` command line tool, follow the steps at ![Installation steps using istioctl](https://istio.io/latest/docs/setup/install/istioctl/)
2. Deploy prometheus using the file ![prometheus.yaml](./k8s_config_files/prometheus.yaml)
- ```bash
  kubectl apply -f k8s_config_files/prometheus.yaml
  ```
> [!NOTE]
> This ![prometheus.yaml](./k8s_config_files/prometheus.yaml) file should work out-of-the-box with istio, because it's the same file from the addons provided by istio project with some minor changes :
> 
> - Number of replicas is set to 2 to assure high availabilty
> - Added a prometheus service of type NodePort to ensure the continuous connectivity with prometheus API.
> - Changed Prometheus to scrape and aggregate metrics at the service level instead of the pod level to reduce scrape target cardinality and lower RAM usage. (RAM usage plateaus around 3.7 GB and stabilizes at about 1.8 GB)
3. Deploy the benchmark application Online Boutique:
- Create a new namespace named `onlineboutique`
  - ```bash
    kubectl create ns onlineboutique 
    ```
- Label the newly created namespace so that istio can inject the sidecars
  - ```bash
    kubectl label namespace onlineboutique istio-injection=enabled
    ``` 
- Finally, deploy the application using the file ![onlineboutique.yaml](./k8s_config_files/onlineboutique.yaml)
  - ```bash
    kubectl apply -n onlineboutique -f k8s_config_files/onlineboutique.yaml
    ```
4. Deploy locust the load generator via this file ![locust.yaml](./k8s_config_files/locust_files/locust.yaml)
  - ```bash
    kubectl apply -f k8s_config_files/locust_files/locust.yaml
    ```
  > [!NOTE]
  > Locust pod runs two containers, Locust v2.10.2 and a locust-exporter that exposes metrics to (such as latency, avg response time) Prometheus ,
  > The exporter requires Locust v2.10.2, which is why we use that version
  > <br/>
  > Locust load generation is implemented in the ![locustfile.py](./k8s_config_files/locust_files/locustfile.py) file which is also packed within a docker image using the ![Dockerfile](./k8s_config_files/locust_files/Dockerfile),
  > In case of any changes needed you can adjust these files to your need and use your custom docker image by setting this ![line](./k8s_config_files/locust_files/locust.yaml#L36C1-L36C63) with your built image, e.g. `ghcr.io/<org>/locust:vX.Y.Z`.

> [!CAUTION]
> All the yaml files used to deploy the previous stack have a `nodeAffinity` that prevents their deployments from being scheduled on the control-plane node (masternode VM), This may cause issues in some setups; please double-check that the nodeAffinity meets your cluster topology.

### Agent Setup & Training 🧠

1. Go to the directory where you cloned the project
  - ```
    cd path/to/project
    ```
2. Install the required packages listed under the file ![requirements.txt](./requirements.txt) in order to run the framework
  - ```
    pip install -r requirements.txt
    ```
3. Install the `gym_hpa` package using
  - ```
    pip install -e .
    ```
4. Change the directory to where the `run.py` script is:
  - ```
    cd src/policies/run
    ```
5. Finally, launch a training
  - ```
    python run.py --k8s --training --total_steps 1000 --alg a2c
    ```
    
> [!TIP]
> Run `python run.py -h` (or `--help`) to list all the available options and their possible values.

## Collaborators 🤝

<div align="center">
  <table border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td align="center" valign="top" width="120">
      <a href="https://github.com/SlyPex">
        <img src="https://images.weserv.nl/?url=github.com/SlyPex.png&h=80&w=80&fit=cover&mask=circle&maxage=1d" alt="Slimane MEHARZI"/>
        <br />
        <sub><b>Slimane MEHARZI (SlyPex)</b></sub>
      </a>
    </td>
    <td align="center" valign="top" width="120">
      <a href="https://github.com/fellahmohamed">
        <img src="https://images.weserv.nl/?url=github.com/fellahmohamed.png&h=80&w=80&fit=cover&mask=circle&maxage=1d" alt="Mohamed Amine FELLAH"/>
        <br />
        <sub><b>Mohamed Amine FELLAH</b></sub>
      </a>
    </td>
    <td align="center" valign="top" width="120">
      <a href="https://github.com/malkiAbdelhamid/">
        <img src="https://images.weserv.nl/?url=github.com/malkiAbdelhamid.png&h=80&w=80&fit=cover&mask=circle&maxage=1d" alt="Abdelhamid Malki"/>
        <br />
        <sub><b>Abdelhamid MALKI</b></sub>
      </a>
    </td>
  </tr>
  </table>
</div>
