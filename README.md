# DeepForest Multi-Agent System

The DeepForest Multi-Agent System provides ecological image analysis by orchestrating multiple AI agents that work together to understand ecological images. Simply upload an image of a forest, wildlife habitat, or ecological scene, and ask questions in natural language.

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

### 4. Configure the HuggingFace Token
Create a `.env` file in the root directory of the deepforest-agent project and add your HuggingFace token like below:

```bash
HF_TOKEN="your_huggingface_token_here"
```

You can obtain your token from [HuggingFace Access Token](https://huggingface.co/settings/tokens). Make sure the Token type is "Write".

## Usage

The DeepForest Agent runs through a Gradio web interface. To start the interface, execute:

```bash
python app.py
```

A link like http://127.0.0.1:7860 will appear in the terminal. Open it in your browser to interact with the agent. A public Gradio link may also be provided if available.

**Sample Recording of Running the System:** [Drive Link](https://drive.google.com/file/d/1gNMn-xJd48Ld3TZU4oiYvTbiWaiLsc8G/view?usp=sharing)


### How to Use

1. Upload an ecological image (aerial/drone photography works best)
2. Ask questions about wildlife, forest health, or ecological patterns. For example:
    - How many trees are detected, and how many of them are alive vs dead?
    - How many birds are around each dead tree?
    - What objects are in the northwest region of the image?
    - Do any birds overlap with livestock in this image?
    - What percentage of the image is covered by trees vs birds vs livestock?
3. Get comprehensive analysis combining computer vision and ecological insights. The gallery shows the annotated image with objects and the detection monitor presents the summary of DeepForest detection.


## Features

- **Multi-Species Detection**: Automatically detects trees, birds, and livestock using specialized DeepForest models
- **Tree Health Assessment**: Identifies alive and dead trees using DeepForest Tree Detector whenever user asks.
- **Visual Analysis**: Dual analysis of original and annotated images using Qwen2.5-VL-3B-Instruct model
- **Memory Context**: Maintains conversation history for contextual understanding across multiple queries
- **Tiling Image for Visual Agent:** Larger images are tiled and processed individually for the visual agent.
- **R-Tree Spatial Indexing:** Stores DeepForest Results in an R-Tree spatial index structure and use spatial queries to retrieve relevant information and present it to the user.
- **Ecological Insights**: Synthesizes detection data with visual analysis and memory context for comprehensive ecological understanding
- **Streaming Responses**: Real-time updates as each agent processes your query


## Requirements

### Hardware Requirements
- **GPU**: GPU with at least 24GB VRAM (recommended for optimal performance). The system is optimized for GPU execution. Running on CPU will take significantly longer processing times
- **Storage**: At least 35GB free space for model downloads

### API Requirements
- **HuggingFace Token**: Required for model access. 


## Image Processing Times

- **Standard Images**: Most ecological images process within 30 seconds on GPU
- **Large GeoTIFF Files**: Larger geospatial images may require significant time for complete analysis


## Models Used

- **SmolLM3-3B**: For Memory Agent to get context, and for Detector Agent to call the tool with appropriate parameters
- **Qwen2.5-VL-3B-Instruct**: Used in Visual agent for multimodal image-text understanding
- **Llama-3.2-3B-Instruct**: For Ecology agents for text understanding and generation
- **DeepForest Models**: For tree, bird, and livestock detection. Also used for alive/dead tree classification.


## Multi-Agent Workflow

[![](https://mermaid.ink/img/pako:eNqlV9ty4jgQ_RWVt2pexmQI9_hhtwiQK4SLgVyceVBMO7giLEq2kzAk_75tSTYiM9k8LA8UpruPuk93H9tby-cLsBzrUdD1kky79xHBT9ubxSDIbM04XcTkfEUf4Scplf4mx15HAE2AuBDHIY_IN3IehUlIWfgL_0zQ9FNhHEv_jkJyIUKccQpio80dae56Q-EvIU4ETbhwyCBlSVhqP0KUkGsungLGXzJUkegw9d2VwT1vACsuNkT6O8RdcdYfVEvVY-3ck24nW-12RmPSDQX4CWlH8QuIf95N0JPM--0W4jdy6vVeMSV0nHLOSIdijuS8q2FPJeyZN4FEhPAMyr4gXUgQOyPligosKHzOuTiTEedez-eMPxYJ9xldUVI9qGDKeWbuJkqQkDDWoecy9M47CSPKyATiNY9iIC9hsiS6rg6PEnjdZ0gVc8XfyIU3D-MUY9sIsEHg_PTxC0SleX9H14U86nJ7kjKmer6LcVPfx44HKdsn7XJHWt9zw-iRwcfQDl-tGRRzcakzIyUyHA5ITwgu3sjAO6GMPVD_ySHTkEHpmMZIaQ6iYwcyw6t8BtVBmXusCBnRxF8SF0dRB1zJgKEncXBAe9gpGYATuabYI2D5QA6l68jDdB_CCPNHEqQnco5Tmacwkm59k4O-_GuM8xBzlsoB6CzBfyIBFzgUsD7hAkccOQwT-hCyMMnPHMvIyVYVMsYuoY2cco6VX3WJAahiGeyzPym67HruU7g2TyumUZ_lyrOmXp8_Euk7ARrjLGnzVJpnelhKw4htdi1EXpc_fz9Ytn3u_XYolv3ZSs7lMdfeKUSQ0Z8vGJItO6iSwjnS_tfS_2a7c1PLts_DTZ4ODpVa1rMweSO3v63ofi9vdqOoogZhjBW127j-4KeY3X_wqZWyrWTx2Jukkek-QF1lsUMSAfDjIRSLHwz1IE64_5QLpFbIzo6MnYK46WpFcbe_4fpwscDlTyBPu6O1s-u5SHUxoCSMDLnSvl0llbdmzrdq0kfepJRlR9w1zQQchXwBr0g97kaSrsn3Pwka0blyke-DWojxOF8c5w9VfC_OmACjmSlehuu8nrFeg8mOiEwzBCwhirMzPxdW9T2T8a7rjUS21R_D9_VxMtHeei30Xky9fTXFnLVu7v74Ko-pXqLZTug_aO6e4rvIPl3tZn2m6pjPvfwm8EvJkM4g63DCyf6dIN8rvVjXnkLd3Smm_Aki8rBRP_K10nt1o0domoqoKDSTravsh2bEvG3rxblRU3XrzdYL82lAPgDIoY2eQcSy1biLOPYFwq0av7s7rxvGa0Y3xfx-R7oiniEslLTHxuAMUBV2U6e-7-4UlPlfnGxQs9skCBlz_oLDoB6AaelqS1CFelA3Lb3cEqCtbFoucrRWUIeWaZl_GoN7oU0-1MA3TdhnjVcOKsGhabrLgw6DFhyZdfmMxnEXArKSTVGPSObhNj5EYYezy6NWOb8svYSLZOlU1q8fYJ7lcJswqroCpubToP4lzEIL_v_OJ1Z9LhbGJK-AgqNDaFS_ggK1fHu1SaYLnHL5qNFqfYXDjUfTvakpcI78SvPhs9IMNNKzT83GmaYL-9Lu2wP7yh7aI7MtptPcvrbbbfv42O50bNT0PdpNx9HIHo9t1LgPfJo-s5k9R8DrPaJMh67tuvZ0at_c2LitJg2WjW8K4cJyAspisK0ViBXNrq1tBnBvoWyt4N5y8GcEKQaxe-s-ese4NY3uOF9ZTiJSjBQ8fVzmF6lUkW5I8TVkVYALfGkA0eFplFhOrXwoMSxna71azmGzfFArN8qNZr1Rb7RqRw3b2lhOqVKttA6alWat1qgf1hvVZu3dtn7JcxsH5VqzUa81Kq1mvdpoVGwLFpmmDNQrkHwTev8XjGAuiw?type=png)](https://mermaid.live/edit#pako:eNqlV9ty4jgQ_RWVt2pexmQI9_hhtwiQK4SLgVyceVBMO7giLEq2kzAk_75tSTYiM9k8LA8UpruPuk93H9tby-cLsBzrUdD1kky79xHBT9ubxSDIbM04XcTkfEUf4Scplf4mx15HAE2AuBDHIY_IN3IehUlIWfgL_0zQ9FNhHEv_jkJyIUKccQpio80dae56Q-EvIU4ETbhwyCBlSVhqP0KUkGsungLGXzJUkegw9d2VwT1vACsuNkT6O8RdcdYfVEvVY-3ck24nW-12RmPSDQX4CWlH8QuIf95N0JPM--0W4jdy6vVeMSV0nHLOSIdijuS8q2FPJeyZN4FEhPAMyr4gXUgQOyPligosKHzOuTiTEedez-eMPxYJ9xldUVI9qGDKeWbuJkqQkDDWoecy9M47CSPKyATiNY9iIC9hsiS6rg6PEnjdZ0gVc8XfyIU3D-MUY9sIsEHg_PTxC0SleX9H14U86nJ7kjKmer6LcVPfx44HKdsn7XJHWt9zw-iRwcfQDl-tGRRzcakzIyUyHA5ITwgu3sjAO6GMPVD_ySHTkEHpmMZIaQ6iYwcyw6t8BtVBmXusCBnRxF8SF0dRB1zJgKEncXBAe9gpGYATuabYI2D5QA6l68jDdB_CCPNHEqQnco5Tmacwkm59k4O-_GuM8xBzlsoB6CzBfyIBFzgUsD7hAkccOQwT-hCyMMnPHMvIyVYVMsYuoY2cco6VX3WJAahiGeyzPym67HruU7g2TyumUZ_lyrOmXp8_Euk7ARrjLGnzVJpnelhKw4htdi1EXpc_fz9Ytn3u_XYolv3ZSs7lMdfeKUSQ0Z8vGJItO6iSwjnS_tfS_2a7c1PLts_DTZ4ODpVa1rMweSO3v63ofi9vdqOoogZhjBW127j-4KeY3X_wqZWyrWTx2Jukkek-QF1lsUMSAfDjIRSLHwz1IE64_5QLpFbIzo6MnYK46WpFcbe_4fpwscDlTyBPu6O1s-u5SHUxoCSMDLnSvl0llbdmzrdq0kfepJRlR9w1zQQchXwBr0g97kaSrsn3Pwka0blyke-DWojxOF8c5w9VfC_OmACjmSlehuu8nrFeg8mOiEwzBCwhirMzPxdW9T2T8a7rjUS21R_D9_VxMtHeei30Xky9fTXFnLVu7v74Ko-pXqLZTug_aO6e4rvIPl3tZn2m6pjPvfwm8EvJkM4g63DCyf6dIN8rvVjXnkLd3Smm_Aki8rBRP_K10nt1o0domoqoKDSTravsh2bEvG3rxblRU3XrzdYL82lAPgDIoY2eQcSy1biLOPYFwq0av7s7rxvGa0Y3xfx-R7oiniEslLTHxuAMUBV2U6e-7-4UlPlfnGxQs9skCBlz_oLDoB6AaelqS1CFelA3Lb3cEqCtbFoucrRWUIeWaZl_GoN7oU0-1MA3TdhnjVcOKsGhabrLgw6DFhyZdfmMxnEXArKSTVGPSObhNj5EYYezy6NWOb8svYSLZOlU1q8fYJ7lcJswqroCpubToP4lzEIL_v_OJ1Z9LhbGJK-AgqNDaFS_ggK1fHu1SaYLnHL5qNFqfYXDjUfTvakpcI78SvPhs9IMNNKzT83GmaYL-9Lu2wP7yh7aI7MtptPcvrbbbfv42O50bNT0PdpNx9HIHo9t1LgPfJo-s5k9R8DrPaJMh67tuvZ0at_c2LitJg2WjW8K4cJyAspisK0ViBXNrq1tBnBvoWyt4N5y8GcEKQaxe-s-ese4NY3uOF9ZTiJSjBQ8fVzmF6lUkW5I8TVkVYALfGkA0eFplFhOrXwoMSxna71azmGzfFArN8qNZr1Rb7RqRw3b2lhOqVKttA6alWat1qgf1hvVZu3dtn7JcxsH5VqzUa81Kq1mvdpoVGwLFpmmDNQrkHwTev8XjGAuiw)