from __future__ import annotations

import json
import hashlib

from voice_agents.tools.definitions import ToolSpec


def get_weather(city: str) -> str:
    """Return stubbed but realistic-looking weather data for a city."""
    # Use city name hash to produce deterministic but varied results
    h = int(hashlib.md5(city.lower().encode()).hexdigest()[:8], 16)
    temp_f = 45 + (h % 50)  # 45-94 °F
    humidity = 30 + (h % 60)  # 30-89 %
    conditions = ["Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Foggy"]
    condition = conditions[h % len(conditions)]

    data = {
        "city": city,
        "temperature_f": temp_f,
        "temperature_c": round((temp_f - 32) * 5 / 9, 1),
        "humidity_pct": humidity,
        "conditions": condition,
        "wind_mph": 3 + (h % 20),
    }
    return json.dumps(data, indent=2)


WEATHER_TOOL_SPEC = ToolSpec(
    name="get_weather",
    description="Get the current weather for a given city. Returns temperature, conditions, humidity, and wind speed.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'Seattle' or 'New York'",
            }
        },
        "required": ["city"],
    },
    fn=get_weather,
)
