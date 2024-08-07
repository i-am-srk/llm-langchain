# from langchain_community.llms import Ollama
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import ollama

# # testing ollama natively
# response = ollama.chat(model='llama2', messages=[
#     {
#         'role' : 'user',
#         'content' : 'why is the sky blue?',
#     },
# ])
# print(response['message']['content'])

# ----- Generate Embeddings -----

# data ingestion
documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = weaviate.connect_to_local()
# print(client.is_ready())

# Create a new data collection
collection_name = "docs" # Name of the data collection
if client.collections.exists(collection_name):
    client.collections.delete(collection_name)

collection = client.collections.create(
    name = collection_name, 
    properties=[
        Property(name="text", data_type=DataType.TEXT), # Name and data type of the property
    ],
)

# Store each document in a vector embedding database
with collection.batch.dynamic() as batch:
  for i, d in enumerate(documents):
    # Generate embeddings
    response = ollama.embeddings(model = "all-minilm", prompt = d)
    embedding = response["embedding"]
    # Add data object with text and embedding
    batch.add_object(
        properties = {"text" : d},
        vector = embedding,
    )

# ----- Retrieve -----

# An example prompt
prompt = "What animals are llamas related to?"

# Generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  model = "all-minilm",
  prompt = prompt,
)

results = collection.query.near_vector(near_vector = response["embedding"], limit = 1)

data = results.objects[0].properties['text']
print(data)

# ----- Generate -----

prompt_template = f"Using this data: {data}. Respond to this prompt: {prompt}"

# Generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model = "llama2",
  prompt = prompt_template,
)

print(output['response'])