# Create a test file: test_k8s_client.py
from kubernetes import client, config
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    # Try to load the kubeconfig
    config.load_kube_config()

    # Create the client
    apps_v1 = client.AppsV1Api()

    # Print the configuration
    configuration = client.Configuration.get_default_copy()
    print(f"API Server Host: {configuration.host}")
    print(f"API Server: {apps_v1.api_client.configuration.host}")

    # Try to read a deployment
    result = apps_v1.read_namespaced_deployment(
        name="recommendationservice", namespace="onlineboutique"
    )
    print(f"Success: {result.metadata.name}")

except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
