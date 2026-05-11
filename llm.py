import os
import base64
import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from utils import *

def get_llm():
    """
        Initializes and returns a Large Language Model (LLM) using AWS Bedrock.
    """
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)

    return ChatBedrock(
        client=session.client("bedrock-runtime"),
        model_id=BEDROCK_MODEL_ID,
        model_kwargs={
            "temperature": 0.0, # Lower temperature for facts
            "max_tokens": 1000
        }
    )

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_answer(query, context_docs):
    """
        Generates a final answer using the LLM based on text and image context.
    """
    llm = get_llm()

    text_contexts = []
    image_contents = []

    for doc in context_docs:
        # If it's a document with image_path, prepare for vision
        image_path = doc.metadata.get("image_path")
        if image_path and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            image_ext = image_path.split('.')[-1].lower()
            if image_ext == 'jpg': image_ext = 'jpeg'
            
            image_contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{image_ext}",
                    "data": base64_image,
                },
            })
        else:
            text_contexts.append(doc.page_content)

    context_text = "\n\n".join(text_contexts)

    prompt_text = f"""
Strictly use the provided text context and images to answer the question. 
Keep the answer direct, concise, and do not use markdown headings or preamble.
If the answer is missing from the context, reply: "I don't know from the provided context."

Text Context:
{context_text}

Question:
{query}
"""

    # Build multimodal content list
    content = [{"type": "text", "text": prompt_text}]
    content.extend(image_contents)

    message = HumanMessage(content=content)
    response = llm.invoke([message])
    return response.content