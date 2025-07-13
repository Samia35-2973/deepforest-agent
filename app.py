import sys
import os
from pathlib import Path
import time
import json
import gradio as gr

# This allows imports to work when app.py is in root but modules are in src/
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"

if not src_dir.exists():
    raise RuntimeError(f"Source directory not found: {src_dir}")

# Add to Python path if not already there
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print(f"App running from: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Python path includes src: {str(src_dir) in sys.path}")

from deepforest_agent.agents.orchestrator import AgentOrchestrator
from deepforest_agent.utils.state_manager import session_state_manager
from deepforest_agent.utils.image_utils import (
    encode_pil_image_to_base64_url, 
    load_pil_image_from_path,
    get_image_info,
    validate_image_path
)
from deepforest_agent.utils.logging_utils import multi_agent_logger


def upload_image(image_path):
    """
    Handle image upload and initialize a new session for the multi-agent workflow.
    
    This function is triggered when a user uploads an image. It creates a new
    session with isolated state and updates the UI to show the chat interface
    and monitoring components.
    
    Args:
        image_path (str or None): The file path to uploaded image from Gradio
        
    Returns:
        tuple: A tuple containing 9 Gradio component updates:
            - gr.Chatbot: Chat interface (visible/hidden)
            - image: Uploaded image state
            - str: Upload status message
            - gr.Textbox: Message input field (visible/hidden)
            - gr.Button: Send button (visible/hidden)
            - gr.Button: Clear button (visible/hidden)
            - gr.Gallery: Generated images gallery (visible/hidden)
            - str: Monitor text with session information
            - str: Session ID for this user
    """
    if image_path is None:
        return (
            gr.Chatbot(visible=False),
            None,  # uploaded_image_state
            "No image uploaded",
            gr.Textbox(visible=False),
            gr.Button(visible=False),  # send_btn
            gr.Button(visible=False),  # clear_btn
            gr.Gallery(visible=False),
            "No image uploaded",
            None  # session_id
        )

    if not validate_image_path(image_path):
        return (
            gr.Chatbot(visible=False),
            None,
            "Invalid image file or path not accessible",
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False), 
            gr.Gallery(visible=False),
            "Invalid image file for analysis.",
            None
        )

    try:
        pil_image = load_pil_image_from_path(image_path)
        if pil_image is None:
            raise Exception("Failed to load image")
        image_info = get_image_info(image_path)
    except Exception as e:
        return (
            gr.Chatbot(visible=False),
            None,
            f"Error loading image: {str(e)}",
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False), 
            gr.Gallery(visible=False),
            "Error loading image for analysis.",
            None
        )

    # Create new session for this user
    session_id = session_state_manager.create_session(pil_image)
    session_state_manager.set(session_id, "image_file_path", image_path)

    detection_monitor = ""

    multi_agent_logger.log_session_event(
        session_id=session_id,
        event_type="session_created",
        details={
            "image_size": image_info.get("size") if image_info else pil_image.size,
            "image_mode": image_info.get("mode") if image_info else pil_image.mode,
            "image_path": image_path,
            "file_size_bytes": image_info.get("file_size_bytes") if image_info else "unknown"
        }
    )

    return (
        gr.Chatbot(visible=True, value=[]),
        pil_image,
        f"Image uploaded successfully! Size: {pil_image.size}",
        gr.Textbox(visible=True),
        gr.Button(visible=True),  # send_btn
        gr.Button(visible=True),  # clear_btn
        gr.Gallery(visible=True, value=[]),
        detection_monitor,
        session_id  # Return session ID
    )


def process_message_streaming(user_message, chatbot_history, generated_images, detection_monitor, session_id):
    """
    Process user message through the multi-agent workflow with streaming updates.
    
    Args:
        user_message (str): The user's input message
        chatbot_history (list): Current chat history for display
        generated_images (list): List of annotated images in PIL Image objects
        detection_monitor (str): Current detection data monitoring text
        session_id (str): Unique session identifier for this user
        
    Yields:
        tuple: A tuple containing 6 updated components:
            - chatbot_history: Updated conversation history
            - msg_input_clear: Empty string to clear message input field
            - generated_images: Updated list of annotated images
            - detection_monitor: Updated detection data monitor
            - send_btn: Button component with interactive state
            - msg_input: Input field component with interactive state
    """
    if not user_message.strip():
        yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)
        return
    
    # Check if session exists
    if session_id is None or not session_state_manager.session_exists(session_id):
        error_msg = "Session expired or invalid. Please upload an image to start a new session."
        chatbot_history.append({"role": "user", "content": user_message})
        chatbot_history.append({"role": "assistant", "content": error_msg})
        yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)
        return
    
    # Check if image is available in session
    current_image = session_state_manager.get(session_id, "current_image")
    if current_image is None:
        error_msg = "No image found in your session. Please upload an image first."
        chatbot_history.append({"role": "user", "content": user_message})
        chatbot_history.append({"role": "assistant", "content": error_msg})
        yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)
        return
    
    total_execution_start = time.perf_counter()

    multi_agent_logger.log_user_query(
        session_id=session_id,
        user_message=user_message
    )
    
    try:
        if session_state_manager.get(session_id, "first_message", True):
            image_base64_url = encode_pil_image_to_base64_url(current_image)
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64_url},
                    {"type": "text", "text": user_message}
                ]
            }
            session_state_manager.set(session_id, "first_message", False)
        else:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
        
        session_state_manager.add_to_conversation(session_id, user_msg)
        chatbot_history.append({"role": "user", "content": user_message})

        chatbot_history.append({"role": "assistant", "content": "Starting analysis..."})

        yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=False), gr.Textbox(interactive=False)
        
        conversation_history = session_state_manager.get(session_id, "conversation_history", [])
        
        print(f"Session {session_id} - User message: {user_message}")
        
        orchestrator = AgentOrchestrator()

        start_time = time.perf_counter()
        
        try:
            # Process with streaming updates
            final_result = None
            
            for result in orchestrator.process_user_message_streaming(
                user_message=user_message,
                conversation_history=conversation_history,
                session_id=session_id
            ):
                if result["type"] == "progress":
                    chatbot_history[-1] = {"role": "assistant", "content": result["message"]}
                    
                    yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=False), gr.Textbox(interactive=False)
                    
                elif result["type"] == "memory_direct":
                    final_response = result["message"]
                    chatbot_history[-1] = {"role": "assistant", "content": final_response}
                    
                    updated_detection_monitor = result.get("detection_data", "")
                    
                    final_result = result
                    
                    yield chatbot_history, "", generated_images, updated_detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)
                    break
                    
                elif result["type"] == "streaming":
                    # Update the last message with streaming response
                    chatbot_history[-1] = {"role": "assistant", "content": result["message"]}
                    
                    yield chatbot_history, "", generated_images, detection_monitor, gr.Button(interactive=False), gr.Textbox(interactive=False)

                    if result.get("is_complete", False):
                        final_response = result["message"]
                    
                elif result["type"] == "final":
                    final_response = result["message"]
                    chatbot_history[-1] = {"role": "assistant", "content": final_response}

                    final_result = result
                    break
            
            if final_result:
                total_execution_time = time.perf_counter() - total_execution_start

                execution_summary = final_result.get("execution_summary", {})
                agent_results = final_result.get("agent_results", {})
                execution_time = final_result.get("execution_time", 0)

                assistant_msg = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_response}]
                }
                session_state_manager.add_to_conversation(session_id, assistant_msg)

                multi_agent_logger.log_agent_execution(
                    session_id=session_id,
                    agent_name="ecology",
                    agent_input="Final synthesis of all agent outputs",
                    agent_output=final_response,
                    execution_time=total_execution_time
                )

                annotated_image = session_state_manager.get(session_id, "annotated_image")
                if annotated_image:
                    generated_images.append(annotated_image)

                updated_detection_monitor = final_result.get("detection_data", "")
                
                yield chatbot_history, "", generated_images, updated_detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)
                
        finally:
            orchestrator.cleanup_all_agents()
        
    except Exception as e:
        total_execution_time = time.perf_counter() - total_execution_start
        error_msg = f"Workflow error: {str(e)}"
        print(f"MAIN APP ERROR (Session {session_id}): {error_msg}")

        multi_agent_logger.log_error(
            session_id=session_id,
            error_type="app_workflow_error", 
            error_message=f"Workflow failed after {total_execution_time:.2f}s: {str(e)}"
        )

        if chatbot_history and chatbot_history[-1]["role"] == "assistant":
            chatbot_history[-1] = {"role": "assistant", "content": error_msg}
        else:
            chatbot_history.append({"role": "assistant", "content": error_msg})
        
        error_detection_monitor = "ERROR: Workflow failed - no detection data available"
        
        yield chatbot_history, "", generated_images, error_detection_monitor, gr.Button(interactive=True), gr.Textbox(interactive=True)

def clear_chat(session_id):
    """
    Clear chat history and cancel any ongoing processing for the session.

    Args:
        session_id (str): The session identifier to clear. Must correspond to
            an existing active session.

    Returns:
        tuple: A tuple containing 5 updated components:
            - chatbot_history: Empty list clearing chat display
            - generated_images: Empty list clearing image gallery
            - monitor_message: Status message indicating successful clear
                operation and session ID
            - send_btn: Re-enabled send button component
            - msg_input: Re-enabled message input component

    """
    if session_id and session_state_manager.session_exists(session_id):
        session_state_manager.cancel_session(session_id)
        session_state_manager.clear_conversation(session_id)

        multi_agent_logger.log_session_event(
            session_id=session_id,
            event_type="conversation_cleared"
        )
        
        return (
            [],  # chatbot
            [],  # generated_images
            "",
            gr.Button(interactive=True),  # Re-enable send button
            gr.Textbox(interactive=True)   # Re-enable message input
        )
    else:
        return (
            [],  # chatbot
            [],  # generated_images
            "",
            gr.Button(interactive=True),   # Re-enable send button
            gr.Textbox(interactive=True)   # Re-enable message input
        )


def create_interface():
    """
    Create and configure the complete Gradio web interface with streaming support.
    
    Returns:
        gr.Blocks: Complete Gradio application interface
    """

    with gr.Blocks(
        title="DeepForest Multi-Agent System",
        theme=gr.themes.Default(
            spacing_size=gr.themes.sizes.spacing_sm,
            radius_size=gr.themes.sizes.radius_none,
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.lime
        )
    ) as app:

        # Gradio State variables
        uploaded_image_state = gr.State(None)
        generated_images_state = gr.State([])
        session_id_state = gr.State(None)

        gr.Markdown("# DeepForest Multi-Agent System")
        gr.Markdown("*DeepForest with SmolLM3-3B + Qwen-VL-3B-Instruct + Llama 3.2-3B-Instruct*")

        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                image_upload = gr.Image(
                    type="filepath",
                    label="Upload Ecological Image", 
                    height=300
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    value="Upload an image to begin analysis",
                    interactive=False
                )

            # Right column
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Multi-Agent Ecological Analysis",
                    height=400,
                    visible=False,
                    show_copy_button=True,
                    type='messages'
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about wildlife, forest health, ecological patterns...",
                        scale=4,
                        visible=False
                    )
                    send_btn = gr.Button("Analyze", scale=1, visible=False, variant="primary")
                    clear_btn = gr.Button("Clear", scale=1, visible=False)

        with gr.Row():
            generated_images_display = gr.Gallery(
                label="Annotated Images after DeepForest Detection",
                columns=2,
                height=400,
                visible=False,
                show_label=True
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Detection Data Monitor")

                detection_data_monitor = gr.Textbox(
                    label="Detection Data Monitor",
                    value="Upload an image and ask a question to see detection data",
                    interactive=False,
                    show_copy_button=True
                )

        with gr.Row(visible=False) as example_row:
            gr.Markdown("""
            **Multi-agent test questions:**
            - How many trees are detected, and how many of them are alive vs dead?
            - How many birds are around each dead tree?
            - What objects are in the northwest region of the image?
            - Do any birds overlap with livestock in this image?
            - What percentage of the image is covered by trees vs birds vs livestock?
            """)

        # Image upload
        image_upload.change(
            fn=upload_image,
            inputs=[image_upload],
            outputs=[
                chatbot,
                uploaded_image_state,
                upload_status,
                msg_input,
                send_btn,
                clear_btn,
                generated_images_display,
                detection_data_monitor,
                session_id_state
            ]
        ).then(
            fn=lambda: gr.Row(visible=True),
            outputs=[example_row]
        )

        # Send button with streaming
        send_btn.click(
            fn=process_message_streaming,
            inputs=[msg_input, chatbot, generated_images_state, detection_data_monitor, session_id_state],
            outputs=[chatbot, msg_input, generated_images_state, detection_data_monitor, send_btn, msg_input]
        ).then(
            fn=lambda images: images,
            inputs=[generated_images_state],
            outputs=[generated_images_display]
        )

        # Enter key with streaming
        msg_input.submit(
            fn=process_message_streaming,
            inputs=[msg_input, chatbot, generated_images_state, detection_data_monitor, session_id_state],
            outputs=[chatbot, msg_input, generated_images_state, detection_data_monitor, send_btn, msg_input]
        ).then(
            fn=lambda images: images,
            inputs=[generated_images_state],
            outputs=[generated_images_display]
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[session_id_state],
            outputs=[chatbot, generated_images_state, detection_data_monitor, send_btn, msg_input]
        ).then(
            fn=lambda: [],
            outputs=[generated_images_display]
        )

    return app


app = create_interface()

if __name__ == "__main__":
    app.launch(
        share=True,
        debug=True,
        show_error=True,
        max_threads=3
    )