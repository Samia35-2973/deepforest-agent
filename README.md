# DeepForest Agent

This repository contains the DeepForest Agent, an AI-powered system designed for ecological object detection and analysis in images. It combines the power of DeepForest computer vision models with conversational AI capabilities

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/weecology/deepforest-agent.git
cd deepforest-agent
```

### 2. Create and activate a Conda environment

```bash
conda create -n deepforest_agent python=3.12.11
conda activate deepforest_agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure your Google API Key
Create a `.env` file in the root directory of the deepforest-agent project and add your Google API key:

```bash
GOOGLE_API_KEY="your_api_key_here"
```

You can obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

The DeepForest Agent runs through a Gradio web interface. To start the interface, execute:

```bash
python -m deepforest_agent.main
```

A link like http://127.0.0.1:7860 will appear in the terminal. Open it in your browser to interact with the agent. A public Gradio link may also be provided if available.


## Features

- **Hybrid AI Analysis:** The agent combines its native computer vision analysis capabilities with the precise detection power of DeepForest Library, offering a more complete ecological assessment. Gemini 2.0 Flash is used to understand user requests, manage conversation flow to provide these insights with the help of *DeepForest* tool to perform:
    - **Object Detection:** Utilizes DeepForest models to detect a range of ecological objects, including birds, trees, and livestock.
    - **Tree Classification:** Includes a feature for classifying detected trees as "alive" or "dead," providing additional ecological insight.
- **Visual Output:** Generates annotated images with bounding boxes around detected objects, making results easy to interpret.
- **Caching:** Implements a caching mechanism to store and reuse previous detection results, reducing redundant computations for repeat requests with similar images and parameters.

## Workflow

[![](https://mermaid.ink/img/pako:eNplVWtvGjkU_StX_rBqVaABktCMVpUolISUR8qjK-1SIYe5Ae967FnbQ0rT_Pe9tsN0uiA-cO1zjs992DyxjU6RJWxreL6DRX-lgD7dv1ZsadHAUOWFg1cL_OZqMMz4Fl-v2Feo19_DB8JcG54KDcthAv8WaA7rDK0lEPwG99qtDdpcK4tEibofArPnmZgJJbpbVK6RkQVZgl-9LuG9AO8TfCAkjrkiaZNAT2fkCqMfuOF2VzL6gfGRGD2-2VUoErmBsAbiASb4GNkl8WMgDjxRqz0aB07DNEfVHcI4JmWT3-_N2_fNBnTTFO4M7oUuLPTR4cYJraDPHffqfM-F5PcSA74V8fODdZgdtcJOm3acI08wNWIrFJcvKdHRI24dhBa8MEqng-D0uqxhvdU4qz_IahWuA-TmacXI_axswnPcvfG7P1asLwwZr-z_gCGpztAVRoFveWXv66_chdaSyimlZ90Sa8dVKnHtaH29ofVKF2-Dm08EKkt1xw3PKDA2gYk2GZfiO8LP1ZL7KXBHJ_20O13IdG0KtU6PopUjRy82qYCx6d7mOCZnBO5fVlOfYCGdPSHOCvWzsZ48CfYxH2iaU0fNT8XGadPI46-1vv-bwLbiYRKsT6tpw7zIMm4O8Aa6SmnHHVmILX8Dt_PppCRPA_nuJO8iT4lEFabFylnjAP_sL4qvpoPQn2P34FG4HZR1gtTPKfULbLRTytxFmRh8DsGMNP34chhoKfVjvchp9OkChgtSkLjb-YGl405u-ywozElhNBrTU7LX_4TZ3qILrIcw8ie0eaAtwoArNJQvDALyZBoXAbn839CeFLfEDwP-S_XhgmWoaAlZRkgMvoTgD99CYXPJD8d58Vn4y0k0VqO3U6QscabAGsuQ6u9D9uQ1VowSzUg-oZ8KC2c4XZmVeiZaztWfWmdHptHFdseSBy4tRbHRfcHpYc7KVYMqRdPThXIs6TTbraDCkif2jWJ6CM7braum_16edS5r7MCSevNd57zxrnneuuy0rjoXV-3Oc419DwefN1pnVxedi8uLNsGbHZKjaaaxHsc_hfDf8Pwfnqjvcg?type=png)](https://mermaid.live/edit#pako:eNplVWtvGjkU_StX_rBqVaABktCMVpUolISUR8qjK-1SIYe5Ae967FnbQ0rT_Pe9tsN0uiA-cO1zjs992DyxjU6RJWxreL6DRX-lgD7dv1ZsadHAUOWFg1cL_OZqMMz4Fl-v2Feo19_DB8JcG54KDcthAv8WaA7rDK0lEPwG99qtDdpcK4tEibofArPnmZgJJbpbVK6RkQVZgl-9LuG9AO8TfCAkjrkiaZNAT2fkCqMfuOF2VzL6gfGRGD2-2VUoErmBsAbiASb4GNkl8WMgDjxRqz0aB07DNEfVHcI4JmWT3-_N2_fNBnTTFO4M7oUuLPTR4cYJraDPHffqfM-F5PcSA74V8fODdZgdtcJOm3acI08wNWIrFJcvKdHRI24dhBa8MEqng-D0uqxhvdU4qz_IahWuA-TmacXI_axswnPcvfG7P1asLwwZr-z_gCGpztAVRoFveWXv66_chdaSyimlZ90Sa8dVKnHtaH29ofVKF2-Dm08EKkt1xw3PKDA2gYk2GZfiO8LP1ZL7KXBHJ_20O13IdG0KtU6PopUjRy82qYCx6d7mOCZnBO5fVlOfYCGdPSHOCvWzsZ48CfYxH2iaU0fNT8XGadPI46-1vv-bwLbiYRKsT6tpw7zIMm4O8Aa6SmnHHVmILX8Dt_PppCRPA_nuJO8iT4lEFabFylnjAP_sL4qvpoPQn2P34FG4HZR1gtTPKfULbLRTytxFmRh8DsGMNP34chhoKfVjvchp9OkChgtSkLjb-YGl405u-ywozElhNBrTU7LX_4TZ3qILrIcw8ie0eaAtwoArNJQvDALyZBoXAbn839CeFLfEDwP-S_XhgmWoaAlZRkgMvoTgD99CYXPJD8d58Vn4y0k0VqO3U6QscabAGsuQ6u9D9uQ1VowSzUg-oZ8KC2c4XZmVeiZaztWfWmdHptHFdseSBy4tRbHRfcHpYc7KVYMqRdPThXIs6TTbraDCkif2jWJ6CM7braum_16edS5r7MCSevNd57zxrnneuuy0rjoXV-3Oc419DwefN1pnVxedi8uLNsGbHZKjaaaxHsc_hfDf8Pwfnqjvcg)