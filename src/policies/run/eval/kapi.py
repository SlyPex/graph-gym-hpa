from kubernetes import client, config

# Load config (choose the right one for your setup)
config.load_kube_config()  # or config.load_incluster_config()

apps_v1 = client.AppsV1Api()

deployments = apps_v1.list_namespaced_deployment(namespace="onlineboutique")

for deploy in deployments.items:
    print(f"{deploy.metadata.name}: desired={deploy.spec.replicas}, available={deploy.status.available_replicas}")
