from bhive import logger
import pydantic
import inspect

BASE_PROMPT = """
Execute these steps to complete your task:
    1. Provide step by step reasoning first inside <thinking></thinking> tags.
    2. Then write a JSON representation of the model into <json></json> tags.

Here is a complete example below:
class Class(BaseModel):
    num_students: int
    teacher: str
    room: str
    students: list[str]

Example output:
<thinking>
This model requires an integer students and other classroom properties.
</thinking>

<json>
{
    "num_students": 3,
    "teacher": "Mr. Smith",
    "room": "101",
    "students": ["Alice", "Bob", "Charlie"]
}
</json>

Be sure to respect the same data types and exact JSON structure as you are provided.
As a reminder, if setting any properties to None you should use the JSON null value.
"""


def parse(text: str, parsing_model: type[pydantic.BaseModel]) -> pydantic.BaseModel | None:
    """Parses the text and validate into the Pydantic model."""
    try:
        parsed_text = parsing_function(text)
        return parsing_model.model_validate_json(parsed_text)
    except Exception as e:
        logger.error(f"Error parsing structured outputs: {e}")
        return None


def prompt(parsing_model: type[pydantic.BaseModel]) -> str:
    """Generates an additional prompt suffix to encourage Pydantic JSON outputs"""
    model_def = inspect.getsource(parsing_model)
    return BASE_PROMPT + f"Here is the Pydantic model with data types:\n{model_def}"


def parsing_function(text: str) -> str:
    """Extracts the JSON from <json> tags in the model's response"""
    return text.split("<json>")[1].split("</json>")[0].strip()
