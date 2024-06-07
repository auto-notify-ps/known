import pika


class Client:
    def __init__(self, server, queue='default'): 
        self.server, self.queue = server, queue
    
    def connect(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.server))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue)

    def send_msg(self, msg):
        # publish on the queue
        self.channel.basic_publish(
        exchange='', 
        routing_key=self.queue, 
        body=msg,
        )

    def send_buffer(self, buf):
        # publish on the queue
        buf.seek(0)
        self.channel.basic_publish(
        exchange='', 
        routing_key=self.queue, 
        body=buf.read(),
        )

    def disconnect(self): self.connection.close()