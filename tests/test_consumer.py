from confluent_kafka import Consumer
import json


consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'vanguard-test-group',
    'auto.offset.reset': 'earliest'
})


consumer.subscribe(['engine-telemetry'])

print("Listening for telemetry data...")
print("-" * 50)

try:
    while True:

        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            print(f"Consumer Error: {msg.error()}")
            continue

        payload = json.loads(
            msg.value().decode('utf-8')
        )

        print(
            f"Engine {payload['unit_number']} | "
            f"Cycle {payload['cycle']}"
        )

except KeyboardInterrupt:
    print("\nStopping consumer...")

finally:
    consumer.close()