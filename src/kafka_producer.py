from confluent_kafka import Producer
import json
import time
from pathlib import Path

producer = Producer({
    'bootstrap.servers' : 'localhost:9092'
})

base_path = Path(__file__).resolve().parent.parent
file_path = base_path / "data/raw/files/test_FD001.txt"

with open(file_path) as file:
    for line in file:
        value = line.split()

        event = {
            "unit_number": int(value[0]),
            "cycle": int(value[1])
        }

        event["op_setting_1"] = float(value[2])
        event["op_setting_2"] = float(value[3])
        event["op_setting_3"] = float(value[4])

        for i in range(21):
            event[f"sensor_{i+1}"] = float(value[i + 5])

        json_message = json.dumps(event)

        producer.produce(
            "engine-telemetry",
            value=json_message
        )
        time.sleep(0.5)

        producer.poll(0)
        print(
            f"Engine {event['unit_number']} "
            f"Cycle {event['cycle']} sent"
        )

producer.flush()