import streamlit as st
import torch
import os
import shutil
import importlib
from transformers import AutoProcessor
from huggingface_hub import create_repo, upload_large_folder, login

SUPPORTED_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3_5ForConditionalGeneration",
        "label": "Qwen 3.5 (Multimodal / Conditional Generation)",
    },
    "Qwen3VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3VLForConditionalGeneration",
        "label": "Qwen 3 VL (Vision-Language)",
    },
    "Qwen2_5_VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "label": "Qwen 2.5 VL (Vision-Language)",
    },
    "Qwen2VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2VLForConditionalGeneration",
        "label": "Qwen 2 VL (Vision-Language)",
    },
}

def get_model_class(architecture: str):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture '{architecture}'")
    info = SUPPORTED_ARCHITECTURES[architecture]
    module = importlib.import_module(info["module"])
    cls = getattr(module, info["class_name"], None)
    if cls is None:
        raise ImportError(f"Cannot find {info['class_name']} in {info['module']}.")
    return cls

# Set up page config
st.set_page_config(
    page_title="ModelResharder - Transformers Streamlit",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 ModelResharder - Transformers Streamlit")
st.markdown("Download, Reshard, and Upload HuggingFace models with custom shard sizes, all from a beautiful Streamlit UI.")

# Check for GPU
gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2)
    st.success(f"✅ GPU Available: **{gpu_name}** ({gpu_mem} GB)")
else:
    st.warning("⚠️ Running on **CPU only**. Processing large models may be slow or run out of memory.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    
    source_model = st.text_input(
        "Source Model Path", 
        value="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID or local path"
    )
    
    target_repo = st.text_input(
        "Target Repository ID", 
        placeholder="username/my-resharded-model",
        help="The Hugging Face repo where the model will be uploaded"
    )
    
    hf_token = st.text_input(
        "HuggingFace Write Token", 
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxx"
    )
    
    shard_size = st.text_input(
        "Max Shard Size", 
        value="4.4GB",
        help="Maximum size per safetensors shard"
    )
    
    arch_options = list(SUPPORTED_ARCHITECTURES.keys())
    arch_labels = [SUPPORTED_ARCHITECTURES[a]["label"] for a in arch_options]
    
    selected_arch_label = st.selectbox("Model Architecture", arch_labels)
    selected_arch = arch_options[arch_labels.index(selected_arch_label)]
    
    start_btn = st.button("🚀 Reshard & Upload Model", type="primary", use_container_width=True)

with col2:
    st.subheader("Process Logs")
    log_container = st.empty()
    
    if start_btn:
        if not source_model or not target_repo or not hf_token:
            st.error("❌ Source Model, Target Repo, and HuggingFace Token are required.")
        else:
            logs = []
            
            def log(msg):
                logs.append(msg)
                # Update text area dynamically
                log_container.text_area("Live Output", value="\n".join(logs), height=500, max_chars=None, key=f"log_{len(logs)}")
                
            with st.spinner("Processing... Please wait."):
                log("Job started...")
                local_dir = f"_resharded_temp_{os.urandom(4).hex()}"
                
                try:
                    log("Authenticating with HuggingFace...")
                    login(token=hf_token)
                    log("✅ Authentication successful")

                    log(f"Creating / verifying repo: {target_repo}")
                    create_repo(repo_id=target_repo, private=True, exist_ok=True)
                    log(f"✅ Repository ready: {target_repo}")

                    log(f"Loading processor from: {source_model}")
                    processor = AutoProcessor.from_pretrained(source_model, trust_remote_code=True)
                    log("✅ Processor loaded")

                    log(f"Loading model [{selected_arch} -- {selected_arch_label}]")
                    ModelClass = get_model_class(selected_arch)
                    device = "cuda" if gpu_available else "cpu"
                    log(f"Device : {device}")

                    model = ModelClass.from_pretrained(
                        source_model,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                    model.eval()
                    log(f"✅ Model loaded on {device}")

                    if gpu_available:
                        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        log(f"GPU mem used: {mem:.2f} GB")

                    os.makedirs(local_dir, exist_ok=True)

                    log(f"Saving model shards (max_shard_size={shard_size})...")
                    model.save_pretrained(local_dir, max_shard_size=shard_size)
                    processor.save_pretrained(local_dir)
                    log("✅ Model + processor saved locally")

                    all_files = sorted(os.listdir(local_dir))
                    shard_files = [f for f in all_files if f.endswith(".safetensors")]
                    log(f"Total files saved : {len(all_files)}")
                    log(f"Safetensor shards : {len(shard_files)}")
                    for fname in shard_files:
                        sz = os.path.getsize(os.path.join(local_dir, fname)) / (1024 ** 2)
                        log(f"  - {fname} ({sz:.1f} MB)")

                    del model
                    if gpu_available:
                        torch.cuda.empty_cache()
                    log("Model unloaded from memory")

                    log(f"Uploading to {target_repo} (upload_large_folder)...")
                    upload_large_folder(
                        repo_id=target_repo,
                        repo_type="model",
                        folder_path=local_dir,
                        revision="main",
                    )
                    log("✅ Upload completed!")
                    log(f"🎉 DONE! Model live at https://huggingface.co/{target_repo}")
                    st.balloons()

                except Exception as exc:
                    log(f"❌ FAILED: {exc}")
                    st.error(f"Failed: {exc}")
                    
                finally:
                    if os.path.exists(local_dir):
                        try:
                            shutil.rmtree(local_dir)
                            log("🧹 Temporary files cleaned up")
                        except Exception as ce:
                            log(f"⚠️ Cleanup warning: {ce}")
                    if gpu_available:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
