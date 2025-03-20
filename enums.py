from enum import Enum
import json


class TestStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
