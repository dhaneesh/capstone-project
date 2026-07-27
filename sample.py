###
#Things learned:
#Shallow copy vs Deep copy
#Anotations in Python
#Use of * and ** in function arguments
#function getattr()
#function json.loads()
###

import json

# This single JSON string contains an object, an array, and a string
json_data = """
{
    "user_object": {"id": 101, "role": "admin"},
    "skills_array": ["Python", "SQL", "Git"],
    "name_string": "Alex Rivera"
}
"""

# print(json_data)
# print(type(json_data))  # This will print <class 'str'>

data = json.loads(json_data)
print(data['name_string'][0])
print(type(data['name_string']))