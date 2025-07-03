import gradio as gr
import os
from PIL import Image
import io
import json
import pandas as pd
import numpy as np

from deepforest_agent.agents.gemini_agent import GeminiAgent

agent = GeminiAgent()

with gr.Blocks() as app:
    with gr.Row():
        image_box = gr.Image(height=500, type="filepath")
   
    with gr.Row():
        chatbot = gr.Chatbot(
            height=750,
            type="messages"
        )
        image_output = gr.Image(height=500, label="Image Output")

    with gr.Row():   
        text_box = gr.Textbox(
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )

    last_annotated_image = gr.State(None)

    def query_message(history, user_prompt, image_path):
        """
        Process user input and prepare the message for the chatbot.
        This function validates input and updates the conversation history.
        """
        if not user_prompt.strip():
            return "", history, None
        
        user_message = {"role": "user", "content": user_prompt}
        
        updated_history = history + [user_message]
        
        return "", updated_history, None

    def bot_response(history, user_prompt, image_path, current_annotated_image):
        """
        Generate the AI agent's response to user input.
        This function handles the core conversation logic and image processing.
        """
        if not history:
            return history, current_annotated_image, current_annotated_image
            
        try:
            response_text, annotated_image = agent.model_response(history, user_prompt, image_path)
            
            if response_text:
                assistant_message = {"role": "assistant", "content": response_text}
                history.append(assistant_message)

            if annotated_image is not None:
                current_annotated_image = annotated_image
            
            return history, current_annotated_image, current_annotated_image
            
        except Exception as e:
            error_message = {"role": "assistant", "content": f"An error occurred: {str(e)}"}
            history.append(error_message)
            return history, current_annotated_image, current_annotated_image

    btn = gr.Button("Submit")

    btn.click(
        query_message, 
        inputs=[chatbot, text_box, image_box], 
        outputs=[text_box, chatbot, image_output]
    ).then(
        bot_response,
        inputs=[chatbot, text_box, image_box, last_annotated_image],
        outputs=[chatbot, image_output, last_annotated_image]
    )

    text_box.submit(
        query_message,
        inputs=[chatbot, text_box, image_box],
        outputs=[text_box, chatbot, image_output]
    ).then(
        bot_response,
        inputs=[chatbot, text_box, image_box, last_annotated_image],
        outputs=[chatbot, image_output, last_annotated_image]
    )

app.queue()
app.launch(share=True, debug=True, show_error=True)