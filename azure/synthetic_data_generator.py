from azure.ai.evaluation import Simulator
import json
import random

def create_synthetic_qa_data():
    """Generate synthetic Q&A data for evaluation"""
    
    # Define topics and questions
    topics = {
        "science": [
            "What is the process of photosynthesis?",
            "How do vaccines work?",
            "What causes climate change?",
            "How do earthquakes occur?",
            "What is the structure of DNA?"
        ],
        "history": [
            "Who was the first president of the United States?",
            "What caused World War II?",
            "When did the Industrial Revolution begin?",
            "Who discovered America?",
            "What was the Cold War?"
        ],
        "technology": [
            "How does machine learning work?",
            "What is blockchain technology?",
            "How do solar panels work?",
            "What is artificial intelligence?",
            "How does the internet work?"
        ]
    }
    
    # Generate synthetic responses and contexts
    synthetic_data = []
    
    for topic, questions in topics.items():
        for question in questions:
            # Generate context based on topic
            context = generate_context(topic, question)
            
            # Generate response
            response = generate_response(question, context)
            
            # Generate ground truth
            ground_truth = generate_ground_truth(question)
            
            synthetic_data.append({
                "query": question,
                "response": response,
                "context": context,
                "ground_truth": ground_truth,
                "topic": topic
            })
    
    return synthetic_data

def generate_context(topic, question):
    """Generate synthetic context based on topic and question"""
    
    contexts = {
        "science": {
            "photosynthesis": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth.",
            "vaccines": "Vaccines work by introducing a weakened or inactive form of a pathogen to stimulate the immune system to produce antibodies. This provides immunity against future infections.",
            "climate_change": "Climate change is primarily caused by the increase in greenhouse gases like carbon dioxide in the atmosphere, largely due to human activities such as burning fossil fuels and deforestation.",
            "earthquakes": "Earthquakes occur when tectonic plates move and release energy in the form of seismic waves. This movement can happen along fault lines where plates meet.",
            "dna": "DNA (deoxyribonucleic acid) is a double-helix molecule that carries genetic information. It consists of four nucleotide bases: adenine, thymine, cytosine, and guanine."
        },
        "history": {
            "president": "George Washington was the first president of the United States, serving from 1789 to 1797. He was a key figure in the American Revolutionary War and helped establish the new nation.",
            "wwii": "World War II was caused by various factors including the Treaty of Versailles, the rise of fascism in Germany and Italy, and territorial expansion by Axis powers.",
            "industrial_revolution": "The Industrial Revolution began in Britain in the late 18th century and was characterized by the transition from manual labor to machine-based manufacturing.",
            "america": "Christopher Columbus is credited with discovering America in 1492, though indigenous peoples had lived there for thousands of years.",
            "cold_war": "The Cold War was a period of geopolitical tension between the United States and the Soviet Union from 1947 to 1991, characterized by proxy wars and nuclear arms race."
        },
        "technology": {
            "machine_learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "blockchain": "Blockchain is a decentralized digital ledger that records transactions across multiple computers securely and transparently.",
            "solar_panels": "Solar panels work by converting sunlight into electricity through photovoltaic cells made of semiconductor materials like silicon.",
            "artificial_intelligence": "Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "internet": "The internet is a global network of interconnected computers that allows for the exchange of information and communication worldwide."
        }
    }
    
    # Find matching context
    for key, context in contexts[topic].items():
        if any(word in question.lower() for word in key.split('_')):
            return context
    
    return f"Context about {topic} and {question}"

def generate_response(question, context):
    """Generate synthetic response based on question and context"""
    
    # Simple response generation based on question type
    if "what" in question.lower():
        return f"Based on the information provided, {context.split('.')[0]}."
    elif "how" in question.lower():
        return f"The process works as follows: {context.split('.')[0]}."
    elif "who" in question.lower():
        return f"The person responsible is mentioned in the context: {context.split('.')[0]}."
    elif "when" in question.lower():
        return f"The timing is described in the context: {context.split('.')[0]}."
    else:
        return f"The answer is: {context.split('.')[0]}."

def generate_ground_truth(question):
    """Generate ground truth answer"""
    
    # Simplified ground truth generation
    ground_truths = {
        "photosynthesis": "Plants convert sunlight, CO2, and water into glucose and oxygen",
        "vaccines": "They stimulate the immune system to produce antibodies",
        "climate_change": "Increase in greenhouse gases from human activities",
        "earthquakes": "Movement of tectonic plates releasing seismic energy",
        "dna": "Double-helix molecule carrying genetic information",
        "president": "George Washington",
        "wwii": "Treaty of Versailles and rise of fascism",
        "industrial_revolution": "Late 18th century transition to machine manufacturing",
        "america": "Christopher Columbus in 1492",
        "cold_war": "US-Soviet tension from 1947-1991",
        "machine_learning": "Computers learning from experience without explicit programming",
        "blockchain": "Decentralized digital ledger for secure transactions",
        "solar_panels": "Convert sunlight to electricity via photovoltaic cells",
        "artificial_intelligence": "Simulation of human intelligence in machines",
        "internet": "Global network of interconnected computers"
    }
    
    for key, truth in ground_truths.items():
        if key in question.lower():
            return truth
    
    return "Answer based on available information"

def create_conversation_data():
    """Generate synthetic conversation data"""
    
    conversations = [
        {
            "messages": [
                {
                    "content": "What is the weather like today?",
                    "role": "user"
                },
                {
                    "content": "The weather is sunny with a high of 75°F.",
                    "role": "assistant",
                    "context": "Current weather data shows sunny conditions with temperature reaching 75°F."
                },
                {
                    "content": "Will it rain tomorrow?",
                    "role": "user"
                },
                {
                    "content": "There's a 30% chance of rain tomorrow.",
                    "role": "assistant",
                    "context": "Weather forecast indicates 30% probability of precipitation tomorrow."
                }
            ]
        },
        {
            "messages": [
                {
                    "content": "How do I make pasta?",
                    "role": "user"
                },
                {
                    "content": "To make pasta, boil water, add salt, cook pasta until al dente, then drain.",
                    "role": "assistant",
                    "context": "Basic pasta cooking instructions: boil salted water, cook until al dente, drain."
                },
                {
                    "content": "How long should I cook it?",
                    "role": "user"
                },
                {
                    "content": "Most pasta takes 8-12 minutes to cook until al dente.",
                    "role": "assistant",
                    "context": "Standard cooking time for pasta is 8-12 minutes for al dente texture."
                }
            ]
        }
    ]
    
    return conversations

def save_synthetic_data():
    """Save synthetic data to JSONL files"""
    
    # Generate Q&A data
    qa_data = create_synthetic_qa_data()
    
    # Save to JSONL file
    with open("synthetic_qa_data.jsonl", "w") as f:
        for item in qa_data:
            f.write(json.dumps(item) + "\n")
    
    # Generate conversation data
    conversation_data = create_conversation_data()
    
    # Save to JSONL file
    with open("synthetic_conversation_data.jsonl", "w") as f:
        for item in conversation_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Generated {len(qa_data)} Q&A samples")
    print(f"Generated {len(conversation_data)} conversation samples")
    print("Data saved to synthetic_qa_data.jsonl and synthetic_conversation_data.jsonl")

def use_azure_simulator():
    """Example of using Azure AI Simulator for data generation"""
    
    # Initialize simulator (requires Azure AI configuration)
    # simulator = Simulator(
    #     azure_ai_project=azure_ai_config,
    #     model_config=model_config
    # )
    
    # Example prompts for data generation
    prompts = [
        "Generate a question about renewable energy",
        "Create a response about climate change",
        "Write a context about machine learning"
    ]
    
    print("Azure AI Simulator example prompts:")
    for prompt in prompts:
        print(f"- {prompt}")
    
    print("\nNote: To use Azure AI Simulator, you need to:")
    print("1. Set up Azure AI project configuration")
    print("2. Configure model settings")
    print("3. Use the Simulator class with proper authentication")

if __name__ == "__main__":
    print("Generating Synthetic Evaluation Data...")
    
    # Generate and save synthetic data
    print("\n1. Generating Q&A and Conversation Data:")
    save_synthetic_data()
    
    # Show Azure Simulator example
    print("\n2. Azure AI Simulator Example:")
    use_azure_simulator()
    
    print("\nSynthetic data generation completed!")
    print("You can now use these files with the evaluation scripts.") 