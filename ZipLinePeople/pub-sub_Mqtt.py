import paho.mqtt.client as mqtt # type: ignore

# MQTT configuration
broker = "broker.hivemq.com"  # Replace with your MQTT broker address
port = 1883  # Replace with your MQTT broker port
topic = "zipline/detected"

# Define the MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    # print(f"Connected with result code {rc}")
    client.subscribe(topic)  # Subscribe to the topic when connected

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} -> {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Disconnected successfully.")

# Initialize MQTT client and set up callbacks
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# Connect to the broker
client.connect(broker, port, 60)

# # Function to publish a message
# def publish_message(message):
#     client.publish(topic, message)
#     print(f"Published message: {message}")
#
# # Run the client loop in a separate thread
# client.loop_start()
#
# # Example usage: Publish a message and keep running
# try:
#     while True:
#         message = input("Enter message to publish (or 'exit' to quit): \n")
#         if message.lower() == 'exit':
#             break
#         publish_message(message)
# except KeyboardInterrupt:
#     print("Exiting...")
#
# # Stop the client loop and disconnect
# client.loop_stop()
client.loop_forever()
client.disconnect()

# import paho.mqtt.client as mqtt
# import time
#
# # MQTT Settings
# broker_address = "broker.hivemq.com"  # Replace with your MQTT broker address
# mqtt_topic = "zipline/detected"
#
# # The callback for when the client receives a CONNACK response from the server
# def on_connect(client, userdata, flags, rc):
#     print(f"Connected to MQTT Broker with result code {rc}")
#     client.subscribe(mqtt_topic)
#
# # The callback for when a message is received from the server
# def on_message(client, userdata, msg):
#     print(f"Message received on {msg.topic}: {msg.payload.decode()}")
#
# # MQTT client initialization
# client = mqtt.Client("Test_Client")
# client.on_connect = on_connect
# client.on_message = on_message
#
# # Connect to the MQTT broker
# try:
#     client.connect(broker_address)
# except Exception as e:
#     print(f"Could not connect to MQTT broker: {e}")
#     exit(1)
#
# # Loop and handle disconnects gracefully
# try:
#     print("Starting MQTT loop. Press Ctrl+C to exit.")
#     client.loop_forever()  # This keeps the connection open and listens for messages
# except KeyboardInterrupt:
#     print("Keyboard Interrupt detected. Disconnecting from MQTT broker...")
# finally:
#     client.disconnect()  # Cleanly disconnect
#     print("Disconnected from MQTT broker. Exiting...")
