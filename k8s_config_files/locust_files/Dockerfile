FROM locustio/locust:2.10.2
WORKDIR /app
COPY locustfile.py .
USER root
ENTRYPOINT [ "locust", "-f" , "locustfile.py", "--host", "http://frontend.onlineboutique.svc.cluster.local", "--autostart", "--web-host", "0.0.0.0"]
