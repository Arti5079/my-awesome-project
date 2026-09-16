import paho.mqtt.client as mqtt
import random
import time

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883

TOPIC_TEMP = "aquirism/sensor/temperature"
TOPIC_PH = "aquirism/sensor/ph"
TOPIC_TDS = "aquirism/sensor/tds"

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("🚀 Starting MQTT Test Publisher... Sending data to broker.emqx.io")

try:
    while True:
        # Generate some realistic random values within normal/abnormal ranges
        temp = round(random.uniform(22.0, 28.5), 1)
        ph = round(random.uniform(6.8, 7.8), 1)
        tds = random.randint(150, 250)

        client.publish(TOPIC_TEMP, str(temp))
        client.publish(TOPIC_PH, str(ph))
        client.publish(TOPIC_TDS, str(tds))

        print(f"Published -> Temp: {temp}°C | pH: {ph} | TDS: {tds} ppm")
        time.sleep(2)  # Send data every 2 seconds

except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()
    print("Publisher stopped.")
