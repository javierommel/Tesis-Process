from langchain.memory import ConversationBufferMemory
import json

# Crear un diccionario para almacenar la memoria de cada usuario
user_memories = {}

def get_user_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory()
    return user_memories[user_id]

def serialize_memory(memory):
    return json.dumps(memory.get())

def deserialize_memory(json_str):
    data = json.loads(json_str)
    memory = ConversationBufferMemory()
    memory.set(data)
    return memory

def clear_user_memory(request):
    user_id = request.form.get('user', '')
    if user_id in user_memories:
        del user_memories[user_id]
